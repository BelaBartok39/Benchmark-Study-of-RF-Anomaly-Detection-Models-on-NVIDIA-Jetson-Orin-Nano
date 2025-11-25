Experimental Methodology
Sustained Inference Performance Evaluation
We conducted rigorous sustained inference experiments to evaluate the real-world deployment performance of RF signal anomaly detection models under continuous operation conditions. The experimental design prioritizes academic rigor and reproducibility, implementing standardized protocols for edge AI performance assessment.

Experimental Configuration
Sustained Inference Protocol: Each model underwent continuous inference testing for 300 seconds (5 minutes) per framework, representing realistic edge deployment scenarios where models must maintain consistent performance over extended periods. This duration ensures thermal equilibrium and captures performance variations that occur during sustained operation.

Temporal Analysis Windows: Performance metrics were calculated using 30-second moving average windows, providing granular temporal analysis while smoothing short-term variations. This windowing approach enables the detection of performance degradation patterns and thermal throttling effects that are critical for edge deployment reliability.

Thermal Management: A 60-second thermal cooldown period was enforced between model evaluations to ensure thermal equilibrium and prevent heat accumulation from influencing subsequent measurements. During cooldown, system power consumption was monitored to verify return to baseline thermal conditions.

Data Sampling Strategy: Experiments utilized 2,000 representative samples (1,000 clean and 1,000 jammed signals) from the 5G dataset, ensuring balanced class representation while maintaining computational feasibility for sustained testing.

Batch Processing: Single-sample inference (batch size = 1) was employed to simulate real-time edge deployment conditions where individual signals must be processed with minimal latency.

Multi-Framework Evaluation
PyTorch Native Implementation: Baseline measurements were conducted using standard PyTorch inference with CUDA acceleration, representing the reference implementation for comparison.

TensorRT Optimization: Models were automatically converted to TensorRT engines during the experimental pipeline, enabling hardware-accelerated inference optimization. The conversion process includes ONNX intermediate representation and TensorRT engine compilation with workspace optimization.

Comprehensive Metrics Collection
Performance Metrics:

Inference latency (milliseconds) with sub-millisecond precision
Throughput (frames per second) calculated from sustained operation
Latency distribution analysis across temporal windows
Power and Energy Efficiency:

High-frequency power monitoring (10 Hz sampling) throughout sustained operation
Average and peak power consumption measurement
Energy efficiency calculation (FPS/Watt) for edge deployment assessment
Energy per inference calculation for battery life estimation
System Resource Utilization:

GPU memory allocation and peak usage tracking
System memory consumption monitoring
CPU and GPU utilization percentages during sustained operation
Thermal Stability Analysis:

Power consumption variability assessment
Performance consistency evaluation using coefficient of variation
Thermal throttling detection through performance degradation analysis
Academic Rigor and Reproducibility
Statistical Validation: Multiple temporal windows provide statistical robustness, with each 30-second window serving as an independent measurement point for variance analysis.

Controlled Environment: Thermal cooldown periods ensure each model evaluation begins from equivalent baseline conditions, eliminating thermal bias between comparisons.

Comprehensive Data Collection: The experimental framework captures time-series data enabling post-hoc analysis of performance patterns, thermal effects, and optimization opportunities.

Hardware Consistency: All experiments were conducted on identical hardware configurations with standardized CUDA and TensorRT environments to ensure reproducible results.

This experimental methodology provides the rigorous foundation necessary for academic publication, enabling comprehensive analysis of edge AI deployment characteristics under realistic operational conditions. The sustained inference approach distinguishes this work from traditional benchmark studies by capturing real-world performance behaviors that emerge only during continuous operation.