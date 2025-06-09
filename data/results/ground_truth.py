#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能捡网球机器人目标检测算法 - 标准结果解析

本模块用于解析标准检测结果，为评估提供参考。
"""

import os
import json

# 标准检测结果字典
GROUND_TRUTH = {}

def load_ground_truth(file_path='xxx.txt')://自行修改图片输出结果的文件名称
    """
    加载标准检测结果
    
    Args:
        file_path: 标准结果文件路径
        
    Returns:
        标准检测结果字典
    """
    global GROUND_TRUTH
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建标准结果文件的完整路径
    full_path = os.path.join(current_dir, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        print(f"警告: 标准结果文件 {full_path} 不存在")
        return {}
    
    try:
        # 读取标准结果文件
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析JSON格式
        GROUND_TRUTH = json.loads(content)
        
        # 处理文件名，确保与实际图像文件名匹配
        processed_gt = {}
        for key, value in GROUND_TRUTH.items():
            # 提取文件名部分
            file_name = os.path.basename(key)
            processed_gt[file_name] = value
        
        GROUND_TRUTH = processed_gt
        
        return GROUND_TRUTH
    except Exception as e:
        print(f"错误: 解析标准结果文件失败: {e}")
        return {}

def get_ground_truth(image_name):
    """
    获取指定图像的标准检测结果
    
    Args:
        image_name: 图像文件名
        
    Returns:
        标准检测结果列表
    """
    global GROUND_TRUTH
    
    # 如果标准结果字典为空，则加载标准结果
    if not GROUND_TRUTH:
        load_ground_truth()
    
    # 返回指定图像的标准检测结果
    return GROUND_TRUTH.get(image_name, [])

# 初始化时加载标准结果
load_ground_truth()
