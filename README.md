# Automatic-tennis-ball-picking-recognition

## 项目概述

本项目为赛题三（智能捡网球机器人）的解决方案。项目基于传统视觉方法和深度学习技术，实现了一套面向端侧高效部署的网球检测系统，可在香橙派-OrangePi AIpro(20T)硬件平台上高效运行。

## 硬件平台

本项目针对以下硬件平台进行优化：
- 香橙派-OrangePi AIpro(20T)
  - 搭载昇腾达芬奇V300 NPU，提供20 TOPS的INT8算力
  - [官网链接](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html)

## 项目结构

```
├── src/             # 项目代码
│   ├── process.py   # 网球检测算法实现
│   ├── main.py      # 完整系统流程
│   └── evaluate.py  # 评估脚本
├── doc/             # 技术文档
│   ├── 实验报告.pdf     # 详细技术方案和算法优化过程与分析
├── data/            # 数据目录
│   ├── images/      # 测试图像
│   ├── results/     # 标准结果和评估结果
│   └── evaluation/  # 评估输出
├── run.sh           # 启动脚本
└── README.md        # 说明文档
```

## 功能特点

- 基于传统视觉方法的网球检测算法
  - 多颜色空间和对比度增强
  - Hough圆检测和模板匹配
  - 专门的小球检测策略
- 详细的算法优化尝试报告
- 深度学习方案建议
- 完整的评估框架

## 使用方法

1. 确保已安装所需依赖
   ```
   pip install opencv-python numpy matplotlib
   ```

2. 运行启动脚本：
   ```
   bash run.sh
   ```

3. 评估算法性能：
   ```
   python src/evaluate.py --visualize
   ```

4. 详细使用说明请参考`doc`目录下的实验报告

## 优化结果

经过多轮优化，传统视觉方法在网球检测任务上的表现如下：

- 精确率: 0.0526
- 召回率: 0.1429
- F1分数: 0.0769
- 平均处理时间: 56.10 ms
