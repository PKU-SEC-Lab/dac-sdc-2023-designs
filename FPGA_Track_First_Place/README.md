# SEUer : DAC-SDC 2023 FPGA Track Champion

## Overview

The DAC 2023 System Design Contest focuses on object detection and classification on an embedded GPU or FPGA system. Contestants will receive a training dataset provided by Baidu, and a hidden dataset will be used to evaluate the performance of the designs in terms of accuracy and speed. Contestants will compete to create the best performing design on a Nvidia Jetson Nano GPU or Xilinx Kria KV260 FPGA board. Grand cash awards will be given to the top three teams. The award ceremony will be held at the 2023 IEEE/ACM Design Automation Conference.

## Introduction

This is a repository for a hardware-efficient DNN accelerator on FPGA specialized in multi-object detection and tracking. The design won first place (FPGA) in [the 60th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](https://dac-sdc.github.io/2023/results-fpga/).

Designed by:

> Jingwei Zhang, Chaoyao Shen, Zenan Cui, Xinye Cao, Meng Zhang, Jun Yang

> SEUer Group, Southeast University

[![picture](https://github.com/AiArtisan/dac_sdc_2023_champion/raw/main/ranking.png)]()


## Contributions

1. **DNN Model Optimization**

* We propose a specialized neural network, UltraSpeed, tailored for multi-object detection and autonomous driving applications. UltraSpeed builds upon the optimizations pioneered in [UltraNet](https://github.com/heheda365/ultra_net), the champion design of the 57th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC), and incorporates advanced low-bit model quantization techniques. This synergy enables our network to strike a delicate balance between accuracy and inference speed on FPGA hardware. By leveraging these optimizations, we maximize the hardware potential while preserving inference accuracy.

2. **More Efficient DSP Optimization Method**

* We introduce an innovative unsigned integer DSP packing scheme, **UINT-Packing<sup>[1]</sup>**, surpassing the popular INT-Packing approach in efficiency. By significantly enhancing MAC unit utilization on DSPs, our method boosts overall computational parallelism, resulting in performance gains exceeding 100%.

3. **Non-aligned Bit-width Converter**

* Addressing mismatches between input data stream bit-widths and AXI interface requirements, we propose a novel non-aligned bit-width converter. Unlike conventional methods involving two-step conversions, our approach achieves bit-width conversion in a single step, drastically reducing resource consumption. Leveraging FIFOs and tail pointers for data storage and retrieval, we reduce FIFO depth from 192 bits to 80 bits, thereby substantially lowering logic resource overhead.

## Build the Project

**Generate HLS project by running:**

```shell
vivado_hls ./src/script/script.tcl
```

## References

[1] More details regarding the UINT-Packing and FPGA accelerator design can be found in our [DAC'23 paper](https://ieeexplore.ieee.org/abstract/document/10247773/). If you find UINT-Packing useful, please cite:

```
@inproceedings{zhang2023uint,
  title={Uint-Packing: Multiply Your DNN Accelerator Performance via Unsigned Integer DSP Packing},
  author={Zhang, Jingwei and Zhang, Meng and Cao, Xinye and Li, Guoqing},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```
