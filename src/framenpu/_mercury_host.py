"""The host side of a Mercury request on the native kernels (_native.Host).

Same contract as the NumPy path in mercury_runtime (input checks, hint prefix, stem
columns, grid staging, readback checks, pooled vector heads). ``check`` refuses any
graph or plan this module does not cover, so a session falls back to NumPy instead
of computing something else.
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

    # ------------------------------------------------------------------ inputs
    def prepare(self, feeds):
        """Checked FP16 grid words for the 1..3 crops of a request: [{name: uint16 array}].
        The hint layer reads its weights once for all crops."""
        import numpy as np

        h, nb = self.host, len(feeds)
        if not 1 <= nb <= 3:
            raise ValueError("Mercury requests carry one to three crops")
        kps = np.empty((nb, 42), np.float32)
        flags = np.empty(nb, np.float32)
        out = []
        for k, feed in enumerate(feeds):
            vals = {}
            for name, shape in mercury.INPUTS.items():
                v = np.ascontiguousarray(feed[name], np.float32)
                if v.shape != shape or h.nonfinite(v.reshape(-1)):
                    raise ValueError(f"{name}: finite values of shape {shape} required")
                vals[name] = v.reshape(-1)
            img16 = np.empty(128 * 128, np.uint16)
            if h.to_f16(vals["inputImg"], img16):
                raise ValueError("image exceeds FP16 storage")
            cols = np.empty(10 * 4096, np.uint16)
            h.stem_columns(img16, cols)
            c, idx, nwords, _ = self.grids[L.STEM_COLS]
            buf = np.empty(c * nwords, np.uint16)
            h.grid_pack(cols, c, idx, nwords, buf)
            out.append({L.STEM_COLS: buf})
            kps[k] = vals["lastKeypoints"]
            flags[k] = vals["useLastKeypoints"][0]
        hint16 = np.empty((nb, 8192), np.uint16)
        tmp = np.empty(nb * 8192, np.float32)
        acc = np.empty(nb * 8192, np.float64)
        if h.hint(self.hint_w, self.hint_b, kps, flags, self.hint_relu6, acc, tmp, hint16):
            raise ValueError("hint exceeds FP16 storage")
        c, idx, nwords, _ = self.grids[L.HINT]
        for k in range(nb):
            buf = np.empty(c * nwords, np.uint16)
            h.grid_pack(hint16[k], c, idx, nwords, buf)
            out[k][L.HINT] = buf
        return out

    def stage(self, mem, prepared):
        for k, bufs in enumerate(prepared):
            where = self.mplan.crop(k).where
            for name, buf in bufs.items():
                at = where[name]
                mem[at : at + buf.nbytes] = memoryview(buf).cast("B")

    # ------------------------------------------------------------------ outputs
    def read(self, get, k):
        """Checked device outputs of crop k: {name: float32 [C, H, W]}; None if invalid."""
        import numpy as np

        where = self.mplan.crop(k).where
        stats = np.zeros(2, np.int32)
        maps = {}
        for name in (L.OUTPUT, L.HEATMAP) if self.heads else (L.OUTPUT,):
            c, idx, nwords, g = self.grids[name]
            raw = np.frombuffer(get(where[name], 2 * c * nwords), np.uint16)
            out = np.empty((c, g.h, g.w), np.float32)
            stray, bad = self.host.grid_unpack(raw, c, idx, nwords, out, stats)
            if stray or bad:
                return None
            maps[name] = out
        return maps

    def finish(self, maps_list):
        """The four Mercury outputs of each crop from its device maps; every tail layer
        reads its weights once for all crops of the request."""
        import numpy as np

        nb, h = len(maps_list), self.host
        def batch(arrays):  # one crop: a view, no copy
            return arrays[0][None] if nb == 1 else np.stack(arrays)

        t = {L.OUTPUT: batch([m[L.OUTPUT] for m in maps_list])}
        if self.heads:
            t["heatmap_xy"] = batch([m[L.HEATMAP][:21] for m in maps_list])
        acc = np.empty(nb * max(op.shape[0] for _, op, _ in self.tail), np.float64)
        for kind, op, extra in self.tail:
            x = t[op.inputs[0]]
            y = np.empty((nb, *op.shape), np.float32)
            if kind == "avgpool":
                kh, kw = extra
                for k in range(nb):
                    h.avgpool(x[k], kh, kw, op.shape[1], op.shape[2], y[k])
            elif kind == "reshape":
                y = x.reshape(nb, *op.shape)
            elif kind == "dense":
                h.dense(extra[0], extra[1], x.reshape(-1), extra[2], acc[: y.size], y)
            else:
                h.conv1x1(extra[0], extra[1], x.reshape(-1), extra[2], acc[: y.size], y)
            t[op.output] = y
        return [
            {
                name: np.ascontiguousarray(t[name][k], np.float32).reshape(shape)
                for name, shape in mercury.OUTPUTS.items()
            }
            for k in range(nb)
        ]
