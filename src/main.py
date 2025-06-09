#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能捡网球机器人目标检测算法 - 主程序

本程序为智能捡网球机器人目标检测算法的主入口，
用于处理输入图像并输出检测结果。
"""

import os
import sys
import json
import time
import argparse
from process import process_img

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='网球检测算法')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出结果保存目录')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化检测结果')
    return parser.parse_args()

def process_single_image(img_path, output_dir, visualize=False):
    """处理单张图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件名
    img_name = os.path.basename(img_path)
    
    # 处理图像
    start_time = time.time()
    result_json = process_img(img_path)
    end_time = time.time()
    
    # 计算处理时间
    process_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 解析结果
    results = json.loads(result_json)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_result.json")
    with open(output_path, 'w') as f:
        f.write(result_json)
    
    print(f"处理图像: {img_path}")
    print(f"检测到 {len(results)} 个网球")
    print(f"处理时间: {process_time:.2f} ms")
    
    # 可视化结果
    if visualize:
        import cv2
        import numpy as np
        
        # 读取原始图像
        img = cv2.imread(img_path)
        
        # 绘制检测结果
        for result in results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 保存可视化结果
        vis_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_vis.jpg")
        cv2.imwrite(vis_path, img)
        
        print(f"可视化结果已保存至: {vis_path}")
    
    return {
        'results': results,
        'process_time': process_time
    }

def process_directory(input_dir, output_dir, visualize=False):
    """处理目录中的所有图像"""
    # 获取目录中的所有图像文件
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        print(f"警告: 目录 {input_dir} 中未找到图像文件")
        return
    
    # 处理每个图像
    total_time = 0
    total_balls = 0
    
    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        result = process_single_image(img_path, output_dir, visualize)
        
        total_time += result['process_time']
        total_balls += len(result['results'])
    
    # 输出统计信息
    avg_time = total_time / len(img_files)
    print("\n统计信息:")
    print(f"总图像数: {len(img_files)}")
    print(f"检测到的网球总数: {total_balls}")
    print(f"平均处理时间: {avg_time:.2f} ms")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入路径 {args.input} 不存在")
        sys.exit(1)
    
    # 处理输入
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.visualize)
    else:
        process_single_image(args.input, args.output, args.visualize)

if __name__ == "__main__":
    main()