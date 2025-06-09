#!/bin/bash

# 智能捡网球机器人目标检测算法启动脚本
# 此脚本用于启动网球检测系统

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH

# 检查是否存在必要的目录和文件
if [ ! -d "./src" ]; then
  echo "错误：未找到src目录"
  exit 1
fi

# 进入src目录
cd ./src

# 运行主程序
echo "正在启动智能捡网球机器人目标检测系统..."
python3 main.py

# 检查运行状态
if [ $? -eq 0 ]; then
  echo "系统启动成功！"
else
  echo "系统启动失败，请检查日志"
  exit 1
fi

exit 0
