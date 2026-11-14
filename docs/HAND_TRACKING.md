# FrameTop landmark backend roadmap

FrameTop defaults to v20a ncnn models. Full convolutional graph execution and ncnn/ONNX conversion remain to be built.

The researched landmark graph has31 Convolution,16 ConvolutionDepthWise,4 InnerProduct,10 BinaryOp,2 Pooling,11 Split, plus shape operations. Fused activations include clip and sigmoid. The inspect-ncnn command produces exact inventory with native_ncnn_loading=false.

Existing pieces: checked model weights; Python submission API; C kernel ABI; HMX GEMM with padding and DSP accumulation; bias/ReLU, clipping and broadcast add; firmware identity checks and observed three-worker completion.

Next work:
1. Introduce a cooperative direct dispatcher and explicit model-memory ownership. The roughly 13 ms benchmark per submission and 50 ms stock lease limit scheduling independently of arithmetic.
2. Add tensor liveness/arena planning for residual branches and persistent weights, preserving gaze/FeatureNN reservations.
3. Decode pinned ncnn param/bin formats, FP16 tags, FP32 biases, padding, activation and broadcast semantics. Reject unsupported layers in preflight.
4. Validate one real landmark convolution against ncnn using im2col/GEMM; optimize packing/transfers with HVX.
5. Add HVX depthwise convolution, pooling, sigmoid and shape/alias operators; validate layers before full-graph inference.
6. Measure crop latency/accuracy, sustained eye tracking coexistence and thermals; add palm detection afterward.

Landmark workload is about 145.6 million MAC/crop; two hands, two views and 30 Hz imply 17.47 GMAC/s before palm search. These are graph estimates, not library benchmarks. The alpha's benchmark-per-tile backend is unsuitable for that frame rate.

Built-ins and custom jobs share the same Device transport, allowing a future dispatcher and kernels to replace the research backend while preserving the developer-facing interface.
