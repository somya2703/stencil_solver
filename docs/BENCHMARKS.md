# Performance Benchmarks

**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  
**CUDA:** 12.8  
**Stencil radius:** 4  

## Kernel comparison (naive vs tiled)

| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | Speedup |
|------|----------------|----------------|---------|
| 64³ | 2101.5 | 1024.0 | **0.49×** |
| 128³ | 1230.3 | 968.7 | **0.79×** |
| 192³ | 1199.1 | 943.3 | **0.79×** |
| 256³ | 1221.1 | 945.4 | **0.77×** |
| 320³ | 1233.4 | 964.0 | **0.78×** |
| 384³ | 1240.0 | 886.4 | **0.71×** |
| 512³ | 1250.8 | 896.6 | **0.72×** |

**Average speedup: 0.72×**

## Plots

![Throughput comparison](plots/throughput_comparison.png)
