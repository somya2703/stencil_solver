# Performance Benchmarks

**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  
**CUDA:** 12.8  
**Stencil radius:** 4  

## Kernel comparison (naive vs tiled)

| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | Speedup |
|------|----------------|----------------|---------|
| 64³ | 234.3 | 115.4 | **0.49×** |
| 128³ | 147.0 | 143.0 | **0.97×** |
| 192³ | 157.4 | 160.1 | **1.02×** |
| 256³ | 174.0 | 164.7 | **0.95×** |
| 320³ | 176.2 | 154.3 | **0.88×** |
| 384³ | 175.7 | 151.9 | **0.86×** |
| 512³ | 178.7 | 156.2 | **0.87×** |

**Average speedup: 0.86×**

## Plots

![Throughput comparison](plots/throughput_comparison.png)
