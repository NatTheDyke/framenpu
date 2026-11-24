# Provenance and third-party materials

framenpu's library, host kernels and build scripts are provided under the repository's MIT
license. The bundled `host_aarch64.so` is built from `src/framenpu/host`; its manifest records
the source hashes.

Model weights are not distributed here. Mercury's keypoint and detector models come from Monado's
[hand-tracking-models](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models)
(no published license; not re-hosted); the students are on Hugging Face under CC BY-NC-SA 4.0
(see [the manifests](src/framenpu/manifests/README.md)).

NumPy, ONNX Runtime, LLVM, Python and GitHub Actions are external, with their own licenses; they
are not vendored. The CDSP backend's provenance is in
[framenpu-cdsp](https://github.com/NatTheDyke/framenpu-cdsp).

This is an independent project, not endorsed by Valve or Qualcomm.
