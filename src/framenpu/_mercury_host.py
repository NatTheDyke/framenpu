"""The model half of a Mercury request on the native kernels (_native.Host): input checks, FP16
conversion, the hint prefix and the pooled vector heads. The layout half (stem columns, grid
staging, readback) is the device backend's: a staging object from the compiled plan
(framenpu_cdsp.staging), which this module only fills and reads by graph tensor name.

Same contract as the NumPy path in mercury_runtime. The constructor refuses any graph or split
this module does not cover, so a session falls back to NumPy instead of computing something else.

Each batch size (1..4 crops) has one preallocated workspace, paired with the staging workspace
of the same size, whose buffers, pointers and kernel calls are bound once, so a request makes no
per-call pointer or size checks. A request is prepare -> stage -> read (per crop) -> finish on
one workspace; callers serialize requests (MercurySession holds its lock throughout).

Covers the Mercury keypoint graph and the 2-view stereo student (inputImg 2x128x128, an
84-value hint, a geom input and the 3D joint head: a 1x1 conv on the FP16 feature map, a
concat and FP32 dense layers); only the ops the requested outputs need run. Mercury's hand
detector (inputImg 1x160x160) has its own request class, NativeDetector.
"""

from . import mercury


class NativeRequest:
    def __init__(self, host, staging, model, split, outputs=None):
        """staging: the plan's native layout half; split: the plan's host/device split."""
        import numpy as np

        self.host, self.staging, self.split = host, staging, split
        self.views = getattr(model, "views", 1)
        self.inputs = dict(getattr(model, "inputs", mercury.INPUTS))
        self.out_shapes = dict(getattr(model, "outputs", mercury.OUTPUTS))
        self.outputs = tuple(self.out_shapes if outputs is None else outputs)
        if self.views not in (1, 2) or self.inputs["inputImg"] != (1, self.views, 128, 128):
            raise ValueError("native staging covers one or two 128x128 views")
        if any(n not in self.out_shapes for n in self.outputs):
            raise ValueError("unknown output")
        if split.staged[:1] != ("inputImg",) or len(split.staged) != 2 or len(split.prefix) != 3:
            raise ValueError("native requests stage the image and the hint plane only")
        hint = split.staged[1]
        gemm, reshape, scale = (model.op(n) for n in split.prefix)
        if (
            gemm.kind != "dense"
            or gemm.inputs != ("lastKeypoints",)
            or gemm.params["weight"].shape != (8192, 42 * self.views)
            or gemm.params["act"] not in (None, "relu6")
            or reshape.kind != "reshape"
            or reshape.inputs != (gemm.output,)
            or scale.kind != "scale"
            or scale.inputs != (reshape.output, "useLastKeypoints")
            or scale.output != hint
            or tuple(scale.shape) != (32, 16, 16)
        ):
            raise ValueError("unexpected Mercury hint prefix")
        # FP32 weights: the kernel widens them, exactly (the model stores FP32)
        if gemm.params["weight"].dtype != np.float32:
            raise ValueError("unexpected hint weight type")
        self.hint_w = np.ascontiguousarray(gemm.params["weight"].T, np.float32)
        self.hint_b = np.ascontiguousarray(gemm.params["bias"], np.float32)
        self.hint_relu6 = gemm.params["act"] == "relu6"

        if getattr(staging, "views", None) != self.views:
            raise ValueError("the staging covers another number of views")
        self._works = {}

        # pooled vector heads: every op after the device part that the DSP does not run and the
        # requested outputs need
        names = [op.name for op in model.ops]
        tail = [op for op in model.ops[names.index(split.last) + 1 :] if op.name not in split.skip]
        want, keep = set(self.outputs), []
        for op in reversed(tail):
            if op.output in want:
                keep.append(op)
                want.update(op.inputs)
        have = set(split.returned) | set(self.inputs)
        missing = want - have - {op.output for op in keep}
        if missing:
            raise ValueError(f"tail needs tensors the request does not produce: {sorted(missing)}")
        self.tail = []
        for op in keep[::-1]:
            p = op.params
            if op.kind == "avgpool":
                kh, kw = p["kernel"]
                self.tail.append(("avgpool", op, (kh, kw)))
            elif op.kind == "reshape":
                self.tail.append(("reshape", op, None))
            elif op.kind == "dense":
                if p["act"] not in (None, "relu6"):
                    raise ValueError(f"{op.name}: unsupported activation")
                if p["weight"].dtype != np.float32:
                    raise ValueError(f"{op.name}: unexpected weight type")
                w = np.ascontiguousarray(p["weight"].T, np.float32)
                b = np.ascontiguousarray(p["bias"], np.float32)
                self.tail.append(("dense", op, (w, b, p["act"] == "relu6")))
            elif (
                op.kind == "conv"
                and len(op.inputs) == 1
                and mercury.unit(op) == "fp32"
                and p["weight"].shape[2:] == (1, 1)
                and p["group"] == 1
                and tuple(p["stride"]) == (1, 1)
                and tuple(p["pads"]) == (0, 0, 0, 0)
                and p["act"] in (None, "relu6")
            ):
                w = np.ascontiguousarray(
                    p["weight"].reshape(p["weight"].shape[0], -1).T, np.float32
                )
                b = np.ascontiguousarray(p["bias"], np.float32)
                self.tail.append(("conv1x1", op, (w, b, p["act"] == "relu6")))
            elif (
                op.kind == "conv"
                and len(op.inputs) == 1
                and mercury.unit(op) == "hmx"
                and p["weight"].shape[2:] == (1, 1)
                and p["group"] == 1
                and tuple(p["stride"]) == (1, 1)
                and tuple(p["pads"]) == (0, 0, 0, 0)
                and p["act"] in (None, "relu6")
            ):
                # the HMX contract: FP16 weights and bias (mercury._conv)
                w = np.ascontiguousarray(
                    mercury._f16(p["weight"].reshape(p["weight"].shape[0], -1)), np.float32
                )
                b = np.ascontiguousarray(mercury._f16(p["bias"]), np.float32)
                self.tail.append(("conv1x1_f16", op, (w, b, p["act"] == "relu6")))
            elif op.kind == "concat" and len(op.shape) == 1:
                self.tail.append(("concat", op, None))
            else:
                raise ValueError(f"{op.name}: no native kernel for {op.kind}")

    # ------------------------------------------------------------------ workspace
    def _work(self, nb, lane=0):
        """The workspace of nb-crop requests; lanes 0..2 let up to three requests of the same
        size be in flight at once (MercurySession.infer_many)."""
        if not 1 <= nb <= 4 or lane not in (0, 1, 2):
            raise ValueError("Mercury requests carry one to four crops")
        w = self._works.get((nb, lane))
        if w is None:
            w = self._works[(nb, lane)] = _Work(self, nb)
        return w

    # ------------------------------------------------------------------ inputs
    def prepare(self, feeds, lane=0):
        """Check and convert the 1..4 crops of a request into their workspace's staged
        FP16 grid words. The hint layer reads its weights once for all crops."""
        import numpy as np

        w = self._work(len(feeds), lane)
        lib, sw = self.host.lib, w.sw
        for k, feed in enumerate(feeds):
            for name, shape in self.inputs.items():
                v = np.asarray(feed[name])
                if v.shape != shape:
                    raise ValueError(f"{name}: finite values of shape {shape} required")
                dst = w.inputs[name][k]
                np.copyto(dst, v.reshape(dst.shape), casting="unsafe")
                if lib.fnpu_nonfinite32(w.input_ptr[name][k], dst.size):
                    raise ValueError(f"{name}: finite values of shape {shape} required")
            if lib.fnpu_to_f16(w.input_ptr["inputImg"][k], sw.img16_p, self.views * 128 * 128):
                raise ValueError("image exceeds FP16 storage")
            sw.pack_image(k)
        if lib.fnpu_hint(*w.hint_args):
            raise ValueError("hint exceeds FP16 storage")
        sw.pack_hint()
        return w

    def stage(self, mem, w):
        w.sw.stage(mem)

    # ------------------------------------------------------------------ outputs
    def read(self, w, get, k):
        """Unpack and check crop k's device outputs into the workspace; False if invalid."""
        return w.sw.read(get, k)

    def load_maps(self, w, maps_list):
        """Put given device maps in the workspace (tests)."""
        w.sw.load_maps(maps_list)

    def finish(self, w):
        """The four Mercury outputs of each crop; every tail layer reads its weights once
        for all crops of the request. The returned arrays are the caller's own."""
        lib = self.host.lib
        for fn, args in w.tail_calls:
            if callable(fn):
                fn(*args)
            else:
                getattr(lib, fn)(*args)
        return [
            {
                name: w.outputs[name][k].reshape(self.out_shapes[name]).copy()
                for name in self.outputs
            }
            for k in range(w.nb)
        ]


class _Work:
    """Buffers, pointers and bound kernel arguments for requests of nb crops."""

    def __init__(self, nr, nb):
        import numpy as np

        def ptr(a):
            if not a.flags.c_contiguous:
                raise ValueError("native workspace arrays must be C-contiguous")
            return a.ctypes.data

        self.nb = nb
        self.sw = sw = nr.staging.work(nb)  # the layout half: FP16 image and hint, device maps
        f32 = np.float32
        self.inputs = {
            name: np.zeros((nb, int(np.prod(shape))), f32) for name, shape in nr.inputs.items()
        }
        self.input_ptr = {n: [ptr(a[k]) for k in range(nb)] for n, a in self.inputs.items()}
        if sw.hint16.shape != (nb, 8192):
            raise ValueError("unexpected staged hint buffer")
        self.hint_tmp = np.zeros(nb * 8192, f32)
        self.hint_acc = np.zeros(nb * 8192, np.float64)
        n_in, n_out = nr.hint_w.shape
        if (n_in, n_out) != (42 * nr.views, 8192):
            raise ValueError("unexpected hint weight shape")
        self.hint_args = (
            ptr(nr.hint_w),
            ptr(nr.hint_b),
            ptr(self.inputs["lastKeypoints"]),
            ptr(self.inputs["useLastKeypoints"]),
            nb,
            n_out,
            n_in,
            int(nr.hint_relu6),
            ptr(self.hint_acc),
            ptr(self.hint_tmp),
            ptr(sw.hint16),
        )

        # the pooled vector heads, bound once, on the returned maps
        t = {name: sw.outputs[name] for name in nr.split.returned}
        for name in nr.inputs.keys() - mercury.INPUTS.keys():  # host-only inputs (geom)
            t[name] = self.inputs[name]
        scratch = np.zeros(
            max(
                [nb * op.shape[0] for _, op, _ in nr.tail]
                + [int(np.prod(op.shape[1:])) for k, op, _ in nr.tail if k == "conv1x1_f16"]
                + [1]
            ),
            np.float64,
        )
        self._keep = [scratch]
        self.tail_calls = []
        for kind, op, extra in nr.tail:
            x = t[op.inputs[0]]
            if kind == "reshape":
                t[op.output] = x.reshape(nb, *op.shape)
                continue
            y = np.zeros((nb, *op.shape), f32)
            self._keep.append(y)
            if kind == "concat":
                parts = [t[i] for i in op.inputs]
                if (
                    any(q.ndim != 2 or q.shape[0] != nb for q in parts)
                    or sum(q.shape[1] for q in parts) != op.shape[0]
                ):
                    raise ValueError(f"{op.name}: unexpected shapes")
                at = 0
                for q in parts:
                    self.tail_calls.append((np.copyto, (y[:, at : at + q.shape[1]], q)))
                    at += q.shape[1]
            elif kind == "conv1x1_f16":
                wt, b, relu = extra
                cout, cin = wt.shape
                if x.shape[1] != cin or not x[0].flags.c_contiguous or op.shape[0] != cout:
                    raise ValueError(f"{op.name}: unexpected shapes")
                hw = int(np.prod(x.shape[2:]))
                for k in range(nb):
                    self.tail_calls.append(
                        (
                            "fnpu_conv1x1_f16",
                            (
                                ptr(wt),
                                ptr(b),
                                ptr(x[k]),
                                cout,
                                cin,
                                hw,
                                int(relu),
                                ptr(scratch),
                                ptr(y[k]),
                            ),
                        )
                    )
            elif kind == "avgpool":
                (kh, kw), (c, h, wd) = extra, x.shape[1:]
                for k in range(nb):
                    self.tail_calls.append(
                        (
                            "fnpu_avgpool",
                            (ptr(x[k]), c, h, wd, kh, kw, op.shape[1], op.shape[2], ptr(y[k])),
                        )
                    )
            else:
                wt, b, relu = extra
                n_in, n_out = wt.shape
                if x[0].size != n_in or y[0].size != n_out:
                    raise ValueError(f"{op.name}: unexpected shapes")
                fn = "fnpu_dense" if kind == "dense" else "fnpu_conv1x1"
                self.tail_calls.append(
                    (
                        fn,
                        (ptr(wt), ptr(b), ptr(x), nb, n_out, n_in, int(relu), ptr(scratch), ptr(y)),
                    )
                )
            t[op.output] = y
        self.outputs = {name: t[name] for name in nr.outputs}


class NativeDetector:
    """The model half of a hand-detector request (mercury.load_detector) on the native kernels;
    same contract as the NumPy path in mercury_runtime (run_tail after the device part): FP16
    conversion by fnpu_to_f16 into the staging workspace (which gathers the stem columns), the
    three dense layers by fnpu_dense (FP64 sums; only their order may differ from NumPy's, by
    one FP32 ulp in rare cases) and the slices and sigmoid on the 8 logits in NumPy, as
    mercury._step computes them. Same workspace scheme as NativeRequest. The constructor
    refuses anything else (the session then uses NumPy)."""

    DENSE = ("Gemm_164", "Gemm_169", "Gemm_174")

    def __init__(self, host, staging, model, split, outputs=None):
        import numpy as np

        self.host, self.staging, self.split = host, staging, split
        if not getattr(model, "detector", False) or dict(model.inputs) != mercury.DET_INPUTS:
            raise ValueError("not the Mercury hand detector")
        self.inputs = dict(model.inputs)
        self.out_shapes = dict(model.outputs)
        self.outputs = tuple(self.out_shapes if outputs is None else outputs)
        if any(n not in self.out_shapes for n in self.outputs):
            raise ValueError("unknown output")
        self.size = mercury.DET_SIZE
        if split.staged != ("inputImg",) or split.prefix or len(split.returned) != 1:
            raise ValueError("native detector requests stage the image of a whole device part")
        if getattr(staging, "size", None) != self.size:
            raise ValueError("the staging covers another image size")
        names = [op.name for op in model.ops]
        tail = model.ops[names.index(split.last) + 1 :]
        kinds = [(op.kind, op.name) for op in tail]
        if [k for k, _ in kinds[:4]] != ["reshape", "dense", "dense", "dense"] or tuple(
            n for _, n in kinds[1:4]
        ) != self.DENSE:
            raise ValueError("unexpected detector tail")
        flat, *dense = tail[:4]
        if flat.inputs != split.returned or tuple(flat.shape) != (1440,):
            raise ValueError("unexpected detector flatten")
        self.dense, x = [], flat.output
        for op in dense:
            p = op.params
            if (
                op.inputs != (x,)
                or p["act"] not in (None, "relu6")
                or p["weight"].dtype != np.float32
            ):
                raise ValueError(f"{op.name}: unexpected dense layer")
            self.dense.append(
                (
                    np.ascontiguousarray(p["weight"].T, np.float32),
                    np.ascontiguousarray(p["bias"], np.float32),
                    p["act"] == "relu6",
                )
            )
            x = op.output
        self.logits = x
        self.heads = []  # (output name, start, end, sigmoid)
        by_out = {op.output: op for op in tail[4:]}
        for name in self.outputs:
            op = by_out.get(name)
            sig = op is not None and op.kind == "sigmoid"
            sl = by_out.get(op.inputs[0]) if sig else op
            if sl is None or sl.kind != "slice" or sl.inputs != (self.logits,):
                raise ValueError(f"{name}: unexpected detector head")
            self.heads.append((name, sl.params["start"], sl.params["end"], sig))
        self._works = {}

    def _work(self, nb, lane=0):
        if not 1 <= nb <= 3 or lane not in (0, 1, 2):
            raise ValueError("Mercury requests carry one to three crops")
        w = self._works.get((nb, lane))
        if w is None:
            w = self._works[(nb, lane)] = _DetWork(self, nb)
        return w

    def prepare(self, feeds, lane=0):
        """Check and convert the 1..3 crops of a request into their staged grid words."""
        import numpy as np

        w = self._work(len(feeds), lane)
        lib, sw = self.host.lib, w.sw
        shape = self.inputs["inputImg"]
        for k, feed in enumerate(feeds):
            v = np.asarray(feed["inputImg"])
            if v.shape != shape:
                raise ValueError(f"inputImg: finite values of shape {shape} required")
            np.copyto(w.img32, v.reshape(-1), casting="unsafe")
            if lib.fnpu_nonfinite32(w.img32_p, w.img32.size):
                raise ValueError(f"inputImg: finite values of shape {shape} required")
            if lib.fnpu_to_f16(w.img32_p, sw.img16_p, w.img32.size):
                raise ValueError("image exceeds FP16 storage")
            sw.pack_image(k)
        return w

    def stage(self, mem, w):
        w.sw.stage(mem)

    def read(self, w, get, k):
        """Unpack and check crop k's final map into the workspace; False if invalid."""
        return w.sw.read(get, k)

    def load_maps(self, w, maps_list):
        """Put given device maps in the workspace (tests)."""
        w.sw.load_maps(maps_list)

    def finish(self, w):
        import numpy as np

        lib = self.host.lib
        for args in w.dense_calls:
            lib.fnpu_dense(*args)
        out = []
        for k in range(w.nb):
            z = w.logits[k]
            o = {}
            for name, a, b, sig in self.heads:
                v = z[a:b]
                if sig:
                    v = (1.0 / (1.0 + np.exp(-v.astype(np.float64)))).astype(np.float32)
                o[name] = np.array(v, np.float32).reshape(self.out_shapes[name])
            out.append(o)
        return out


class _DetWork:
    """Buffers, pointers and bound kernel arguments for detector requests of nb crops."""

    def __init__(self, nd, nb):
        import numpy as np

        def ptr(a):
            if not a.flags.c_contiguous:
                raise ValueError("native workspace arrays must be C-contiguous")
            return a.ctypes.data

        self.nb = nb
        self.sw = sw = nd.staging.work(nb)
        n = nd.size * nd.size
        self.img32 = np.zeros(n, np.float32)
        self.img32_p = ptr(self.img32)
        maps = sw.outputs[nd.split.returned[0]]  # (nb, 1440): flattened = Flatten_163
        self.dense_calls, x, self._keep = [], maps, []
        for wt, b, relu in nd.dense:
            n_in, n_out = wt.shape
            if x.shape[1] != n_in:
                raise ValueError("unexpected detector dense shapes")
            y = np.zeros((nb, n_out), np.float32)
            acc = np.zeros(nb * n_out, np.float64)
            self.dense_calls.append(
                (ptr(wt), ptr(b), ptr(x), nb, n_out, n_in, int(relu), ptr(acc), ptr(y))
            )
            self._keep += [acc, y]
            x = y
        self.logits = x
