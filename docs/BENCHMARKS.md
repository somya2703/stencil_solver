# Performance Benchmarks

**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  
**CUDA:** 12.8  
**Stencil radius:** 4  

## Kernel comparison (naive vs tiled)

| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | Speedup |
|------|----------------|----------------|---------|
| 64³ | 237.2 | 114.5 | **0.48×** |
| 128³ | 147.0 | 123.6 | **0.84×** |
| 192³ | 161.4 | 135.9 | **0.84×** |
| 256³ | 173.9 | 135.4 | **0.78×** |
| 320³ | 176.2 | 137.2 | **0.78×** |
| 384³ | 177.3 | 126.5 | **0.71×** |
| 512³ | 178.9 | 128.2 | **0.72×** |

**Average speedup: 0.74×**

## Plots

![Throughput comparison](plots/throughput_comparison.png)
