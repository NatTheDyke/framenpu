# steam-frame-hand-tracking backend

`steam-frame-hand-tracking.patch` applies to
[spyjacob/steam-frame-hand-tracking](https://github.com/spyjacob/steam-frame-hand-tracking)
at commit `8852ec9bdd4f81fb5a359e5bf88ed25f0e6fc89e`.
It adds optional FrameNPU Mercury inference with CPU default and fallback,
immediate groups of one to three available crops, worker idle/shutdown
cleanup, fresh-frame admission (`--merc-fresh`) and camera-idle power release. It includes the tracker tests and `docs/FRAMENPU.md` setup guide.

Apply it from that tracker checkout with `git apply /path/to/this.patch`.
The matching FrameNPU API, build/deployment steps and validation limits are
in [MERCURY_INTEGRATION.md](../docs/MERCURY_INTEGRATION.md).
Neither weights nor camera data are included.

The committed integration is also available on the
[framenpu-inference branch](https://github.com/NatTheDyke/steam-frame-hand-tracking/tree/framenpu-inference)
of the tracker fork (commit `17c8fa9be1dff61efdcf540d061b5b7472687b80`).
