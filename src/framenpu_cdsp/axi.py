"""Job-scoped CDSP AXI-master clock raise (EXPERIMENTAL, opt-in with FNPU_AXI_MHZ).

While our jobs run, the firmware leaves the Q6 AXI master (turing_cc_q6ss_q6_axim_clk, root CMD
0x32008254 / CFG 0x32008258) on the 19.2 MHz crystal with every source PLL off, which caps the
DSP's DDR traffic near 1.2 GB/s. The firmware's own frequency table for that clock (domain
f023c8e0, records at f023cd78: 366.7 ... 1500 MHz) has a 750 MHz entry: turing_cc_core2_pll
(0x32002000, L 78, alpha 0x2000 = 1500 MHz), mux source 5, divide by 2, rail level 192.

This owner uses integer L 78 (1497.6 MHz, alpha 0), so the AXI master runs at 748.8 MHz, just
under that entry. It acts only under the CDSP operation gate, inside one job: raise() after the
claim and before publishing, restore() after completion and before acknowledging, so outside
our jobs the hardware is exactly as the firmware left it and the firmware's clock driver never
meets a state it did not set. raise() refuses unless the clock is on the crystal and the PLL is
in its reset state; restore() refuses to stop the PLL if the mux no longer selects it.
"""

import mmap
import os
import struct
import time

from .errors import DeviceBusy, RecoveryRequired

RCG_PAGE, PLL_PAGE = 0x32008000, 0x32002000
CMD, CFG = 0x254, 0x258
SRC_MASK = 0x71F                       # CFG source select [10:8] and divider [4:0]
PLL_WORDS = (0, 4, 0x10, 0x14, 0x18)   # mode/lock, opmode, L, alpha, user control
PLL_RESET = {0: 0x100, 4: 0, 0x10: 0, 0x14: 0, 0x18: 1}
# MHz -> (PLL L at 19.2 MHz with alpha 0, CFG source/divider bits)
AXI_RATES = {748.8: (78, 0x503)}


def rate_from_env():
    """The FNPU_AXI_MHZ rate, or None when unset or empty (the default: no AXI change)."""
    value = os.environ.get("FNPU_AXI_MHZ", "")
    if value == "":
        return None
    try:
        mhz = float(value)
    except ValueError:
        raise ValueError("FNPU_AXI_MHZ must be one of %s" % sorted(AXI_RATES)) from None
    if mhz not in AXI_RATES:
        raise ValueError("FNPU_AXI_MHZ must be one of %s" % sorted(AXI_RATES))
    return mhz


class AxiClock:
    def __init__(self, mhz, mmio_factory=None, poll_timeout=0.005):
        if mhz not in AXI_RATES:
            raise ValueError("AXI rate must be one of %s MHz" % sorted(AXI_RATES))
        self.mhz = mhz
        self.pll_l, self.src = AXI_RATES[mhz]
        self._factory = mmio_factory
        self.poll_timeout = poll_timeout
        self.fd = None
        self.rcg = self.pll = None
        self.raised = False
        self.abandoned = False
        self.saved_cfg = None
        self.raises = 0
        self.raise_s = self.restore_s = 0.0

    # --- register access -------------------------------------------------------------------
    def _map(self):
        if self.rcg is not None:
            return
        if self._factory:
            self.rcg, self.pll = self._factory(RCG_PAGE), self._factory(PLL_PAGE)
            return
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.rcg, self.pll = (
            mmap.mmap(self.fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=a)
            for a in (RCG_PAGE, PLL_PAGE)
        )

    @staticmethod
    def _r(m, off):
        return struct.unpack_from("<I", m, off)[0]

    @staticmethod
    def _w(m, off, value):
        struct.pack_into("<I", m, off, value)
        return struct.unpack_from("<I", m, off)[0]

    def _poll(self, m, off, mask, value):
        end = time.monotonic() + self.poll_timeout
        while True:
            if self._r(m, off) & mask == value:
                return True
            if time.monotonic() > end:
                return False

    def _pll_state(self):
        return {o: self._r(self.pll, o) for o in PLL_WORDS}

    def _update(self):
        """Latch CFG: set CMD.UPDATE and wait for the hardware to clear it; the root must stay on."""
        self._w(self.rcg, CMD, self._r(self.rcg, CMD) | 1)
        return self._poll(self.rcg, CMD, 1, 0) and not self._r(self.rcg, CMD) & 0x80000000

    def _pll_off(self):
        """Reset-state PLL words, output first (power.py's HMX close order), then verify."""
        p = self.pll
        self._w(p, 0, self._r(p, 0) & ~1)
        self._w(p, 4, PLL_RESET[4])
        self._w(p, 0, PLL_RESET[0])
        for off in (0x18, 0x14, 0x10):
            self._w(p, off, PLL_RESET[off])
        return self._pll_state() == PLL_RESET

    # --- job scope -------------------------------------------------------------------------
    def raise_(self):
        """Caller holds the operation gate and has not published. Raises DeviceBusy (nothing
        changed) if the clock is not in the firmware's crystal state."""
        if self.abandoned:
            raise RecoveryRequired("AXI clock owner was abandoned")
        if self.raised:
            raise RecoveryRequired("AXI clock is already raised by this owner")
        t0 = time.perf_counter()
        self._map()
        cmd, cfg = self._r(self.rcg, CMD), self._r(self.rcg, CFG)
        if cmd != 0 or cfg & SRC_MASK != 0:
            raise DeviceBusy("AXI master is not on the crystal (CMD %#x CFG %#x)" % (cmd, cfg))
        if self._pll_state() != PLL_RESET:
            raise DeviceBusy("core2 PLL is not in its reset state: %r" % self._pll_state())
        self.saved_cfg = cfg
        p = self.pll
        self._w(p, 0x10, self.pll_l)
        self._w(p, 0x14, 0)
        self._w(p, 4, 1)
        self._w(p, 0, PLL_RESET[0] | 4)
        if not self._poll(p, 0, 1 << 31, 1 << 31):
            ok = self._pll_off()
            if not ok:
                self.abandoned = True
                raise RecoveryRequired("core2 PLL did not lock and did not return to reset")
            raise RuntimeError("core2 PLL did not lock; restored")
        self._w(p, 0, self._r(p, 0) | 1)
        self.raised = True                      # from here on the mux may select the PLL
        self._w(self.rcg, CFG, (cfg & ~SRC_MASK) | self.src)
        if not self._update() or self._r(self.rcg, CFG) & SRC_MASK != self.src:
            self.abandoned = True
            raise RecoveryRequired("AXI root update to %.1f MHz did not complete" % self.mhz)
        self.raises += 1
        self.raise_s += time.perf_counter() - t0

    def restore(self):
        """Back to the saved crystal CFG, then the PLL to reset. Caller still holds the gate;
        the DSP job has completed."""
        if not self.raised:
            return
        t0 = time.perf_counter()
        if self._r(self.rcg, CFG) & SRC_MASK != self.src:
            self.abandoned = True
            raise RecoveryRequired("AXI mux changed while raised; core2 PLL left running")
        self._w(self.rcg, CFG, self.saved_cfg)
        if not self._update() or self._r(self.rcg, CFG) != self.saved_cfg:
            self.abandoned = True
            raise RecoveryRequired("AXI root did not return to the crystal; core2 PLL left running")
        self.raised = False
        if not self._pll_off():
            self.abandoned = True
            raise RecoveryRequired("core2 PLL did not return to its reset state")
        self.restore_s += time.perf_counter() - t0

    def abandon(self):
        """After a job that may still be running: leave every clock as it is."""
        self.abandoned = True
        self.close()

    def close(self):
        for m in (self.rcg, self.pll):
            if m is not None and not self._factory:
                m.close()
        self.rcg = self.pll = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
