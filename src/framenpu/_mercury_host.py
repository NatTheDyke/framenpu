"""The host side of a Mercury request on the native kernels (_native.Host).

Same contract as the NumPy path in mercury_runtime (input checks, hint prefix, stem
columns, grid staging, readback checks, pooled vector heads). The constructor refuses
any graph or plan this module does not cover, so a session falls back to NumPy instead
of computing something else.

Each batch size (1..3 crops) has one preallocated workspace whose buffers, pointers
and kernel calls are bound once, so a request makes no per-call pointer or size
checks. A request is prepare -> stage -> read (per crop) -> finish on one workspace;
callers serialize requests (MercurySession holds its lock throughout).
"""

from . import mercury
from . import _mercury_common as C
from . import _mercury_layout as L


class NativeRequest:
    def __init__(self, host, model, mplan, heads):
        import numpy as np

        self.host, self.heads = host, heads
        mp = mplan.crop(0)
        if list(mp.external + mp.host_parts) != [L.STEM_COLS, L.HINT]:
            raise ValueError("native staging covers the stem columns and the hint plane only")
        gemm, reshape, scale = (model.op(n) for n in L.HOST_HINT_OPS[:3])
        if (
            gemm.kind != "dense"
            or gemm.inputs != ("lastKeypoints",)
            or gemm.params["weight"].shape != (8192, 42)
            or gemm.params["act"] not in (None, "relu6")
            or reshape.kind != "reshape"
            or reshape.inputs != (gemm.output,)
            or scale.kind != "scale"
            or scale.inputs != (reshape.output, "useLastKeypoints")
            or scale.output != L.HINT
            or tuple(scale.shape) != (32, 16, 16)
        ):
            raise ValueError("unexpected Mercury hint prefix")
        # FP32 weights: the kernel widens them, exactly (the model stores FP32)
        if gemm.params["weight"].dtype != np.float32:
            raise ValueError("unexpected hint weight type")
        self.hint_w = np.ascontiguousarray(gemm.params["weight"].T, np.float32)
        self.hint_b = np.ascontiguousarray(gemm.params["bias"], np.float32)
        self.hint_relu6 = gemm.params["act"] == "relu6"

        self.grids = {}
        names = [L.STEM_COLS, L.HINT, L.OUTPUT] + ([L.HEATMAP] if heads else [])
        for name in names:
            g, c = mp.grid(name), mp.shape[name][0]
            idx, _ = C._grid_index(g.w, g.h)
            nwords = g.nbin * 32
            if c % 2 or g.tensor_bytes(c) != 2 * c * nwords:
                raise ValueError(f"unexpected grid for {name}")
            for k in range(len(mplan.crops) or 1):
                ck = mplan.crop(k)
                if ck.grid(name) != g or ck.shape[name] != mp.shape[name]:
                    raise ValueError(f"crop {k} lays out {name} differently")
            self.grids[name] = (c, np.ascontiguousarray(idx, np.int32), nwords, g)
        if self.grids[L.STEM_COLS][0] != 10 or self.grids[L.HINT][0] != 32:
            raise ValueError("unexpected staged channel counts")
        self.mplan = mplan
        self._works = {}

        # pooled vector heads: every op after L.LAST that the DSP does not run
        names = [op.name for op in model.ops]
        skip = L.HEAD_OPS if heads else ()
        self.tail = []
        for op in model.ops[names.index(L.LAST) + 1 :]:
            if op.name in skip:
                continue
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
                w = np.ascontiguousarray(p["weight"].reshape(p["weight"].shape[0], -1).T, np.float32)
                b = np.ascontiguousarray(p["bias"], np.float32)
                self.tail.append(("conv1x1", op, (w, b, p["act"] == "relu6")))
            else:
                raise ValueError(f"{op.name}: no native kernel for {op.kind}")

    # ------------------------------------------------------------------ workspace
    def _work(self, nb):
        if not 1 <= nb <= 3:
            raise ValueError("Mercury requests carry one to three crops")
        w = self._works.get(nb)
        if w is None:
            w = self._works[nb] = _Work(self, nb)
        return w

    # ------------------------------------------------------------------ inputs
    def prepare(self, feeds):
        """Check and convert the 1..3 crops of a request into their workspace's staged
        FP16 grid words. The hint layer reads its weights once for all crops."""
        import numpy as np

        w = self._work(len(feeds))
        lib = self.host.lib
        for k, feed in enumerate(feeds):
            for name, shape in mercury.INPUTS.items():
                v = np.asarray(feed[name])
                if v.shape != shape:
                    raise ValueError(f"{name}: finite values of shape {shape} required")
                dst = w.inputs[name][k]
                np.copyto(dst, v.reshape(dst.shape), casting="unsafe")
                if lib.fnpu_nonfinite32(w.input_ptr[name][k], dst.size):
                    raise ValueError(f"{name}: finite values of shape {shape} required")
            if lib.fnpu_to_f16(w.input_ptr["inputImg"][k], w.img16_p, 128 * 128):
                raise ValueError("image exceeds FP16 storage")
            lib.fnpu_stem_columns(w.img16_p, w.cols_p)
            lib.fnpu_grid_pack(*w.stem_pack[k])
        if lib.fnpu_hint(*w.hint_args):
            raise ValueError("hint exceeds FP16 storage")
        for args in w.hint_pack:
            lib.fnpu_grid_pack(*args)
        return w

    def stage(self, mem, w):
        for k in range(w.nb):
            where = self.mplan.crop(k).where
            for name, words in w.staged[k]:
                at = where[name]
                mem[at : at + len(words)] = words

    # ------------------------------------------------------------------ outputs
    def read(self, w, get, k):
        """Unpack and check crop k's device outputs into the workspace; False if invalid."""
        lib = self.host.lib
        where = self.mplan.crop(k).where
        for name in w.read_names:
            raw = w.raw[name]
            w.raw_bytes[name][:] = get(where[name], raw.nbytes)
            lib.fnpu_grid_unpack(*w.unpack[name][k])
            if w.stats[0] or w.stats[1]:
                return False
        return True

    def load_maps(self, w, maps_list):
        """Put given device maps in the workspace (tests)."""
        for k, maps in enumerate(maps_list):
            for name in w.read_names:
                w.maps[name][k] = maps[name]

    def finish(self, w):
        """The four Mercury outputs of each crop; every tail layer reads its weights once
        for all crops of the request. The returned arrays are the caller's own."""
        lib = self.host.lib
        for fn, args in w.tail_calls:
            getattr(lib, fn)(*args)
        return [
            {name: w.outputs[name][k].reshape(shape).copy() for name, shape in mercury.OUTPUTS.items()}
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
        f32, u16 = np.float32, np.uint16
        self.inputs = {
            name: np.zeros((nb, int(np.prod(shape))), f32) for name, shape in mercury.INPUTS.items()
        }
        self.input_ptr = {n: [ptr(a[k]) for k in range(nb)] for n, a in self.inputs.items()}
        self.img16 = np.zeros(128 * 128, u16)
        self.cols = np.zeros(10 * 4096, u16)
        self.img16_p, self.cols_p = ptr(self.img16), ptr(self.cols)

        c, idx, nwords, _ = nr.grids[L.STEM_COLS]
        self.stem = np.zeros((nb, c * nwords), u16)
        self.stem_pack = [
            (self.cols_p, c, idx.size, ptr(idx), nwords, ptr(self.stem[k])) for k in range(nb)
        ]
        self.hint16 = np.zeros((nb, 8192), u16)
        self.hint_tmp = np.zeros(nb * 8192, f32)
        self.hint_acc = np.zeros(nb * 8192, np.float64)
        n_in, n_out = nr.hint_w.shape
        if (n_in, n_out) != (42, 8192):
            raise ValueError("unexpected hint weight shape")
        self.hint_args = (
            ptr(nr.hint_w), ptr(nr.hint_b), ptr(self.inputs["lastKeypoints"]),
            ptr(self.inputs["useLastKeypoints"]), nb, n_out, n_in, int(nr.hint_relu6),
            ptr(self.hint_acc), ptr(self.hint_tmp), ptr(self.hint16),
        )
        c, idx, nwords, _ = nr.grids[L.HINT]
        self.hint_grid = np.zeros((nb, c * nwords), u16)
        self.hint_pack = [
            (ptr(self.hint16[k]), c, idx.size, ptr(idx), nwords, ptr(self.hint_grid[k]))
            for k in range(nb)
        ]
        self.staged = [
            ((L.STEM_COLS, memoryview(self.stem[k]).cast("B")),
             (L.HINT, memoryview(self.hint_grid[k]).cast("B")))
            for k in range(nb)
        ]

        self.read_names = (L.OUTPUT, L.HEATMAP) if nr.heads else (L.OUTPUT,)
        self.stats = np.zeros(2, np.int32)
        self.raw, self.raw_bytes, self.maps, self.unpack = {}, {}, {}, {}
        for name in self.read_names:
            c, idx, nwords, g = nr.grids[name]
            self.raw[name] = np.zeros(c * nwords, u16)
            self.raw_bytes[name] = memoryview(self.raw[name]).cast("B")
            self.maps[name] = np.zeros((nb, c, g.h, g.w), f32)
            self.unpack[name] = [
                (ptr(self.raw[name]), c, idx.size, ptr(idx), nwords, ptr(self.maps[name][k]),
                 ptr(self.stats))
                for k in range(nb)
            ]

        # the pooled vector heads, bound once
        t = {L.OUTPUT: self.maps[L.OUTPUT]}
        if nr.heads:
            t["heatmap_xy"] = self.maps[L.HEATMAP][:, :21]
        scratch = np.zeros(nb * max(op.shape[0] for _, op, _ in nr.tail), np.float64)
        self._keep = [scratch]
        self.tail_calls = []
        for kind, op, extra in nr.tail:
            x = t[op.inputs[0]]
            if kind == "reshape":
                t[op.output] = x.reshape(nb, *op.shape)
                continue
            y = np.zeros((nb, *op.shape), f32)
            self._keep.append(y)
            if kind == "avgpool":
                (kh, kw), (c, h, wd) = extra, x.shape[1:]
                for k in range(nb):
                    self.tail_calls.append(
                        ("fnpu_avgpool", (ptr(x[k]), c, h, wd, kh, kw, op.shape[1], op.shape[2], ptr(y[k])))
                    )
            else:
                wt, b, relu = extra
                n_in, n_out = wt.shape
                if x[0].size != n_in or y[0].size != n_out:
                    raise ValueError(f"{op.name}: unexpected shapes")
                fn = "fnpu_dense" if kind == "dense" else "fnpu_conv1x1"
                self.tail_calls.append(
                    (fn, (ptr(wt), ptr(b), ptr(x), nb, n_out, n_in, int(relu), ptr(scratch), ptr(y)))
                )
            t[op.output] = y
        self.outputs = {name: t[name] for name in mercury.OUTPUTS}
