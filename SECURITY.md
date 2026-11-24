# Security

`framenpu serve` runs as root and accepts one user's connections over a Unix socket; it parses a
JSON header and float32 payloads of validated shapes, never pickle. The device boundary (trusted
DSP jobs, /dev/mem power control, firmware checks) is in
[framenpu-cdsp](https://github.com/NatTheDyke/framenpu-cdsp)'s SECURITY.md.

Use GitHub's private "Report a vulnerability" feature if enabled. If unavailable, open an issue
requesting a private contact without posting sensitive details. Omit credentials, serial numbers,
raw captures and firmware dumps. The project has not had an independent security audit.
