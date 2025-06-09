#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能捡网球机器人目标检测算法 - 评估脚本

本脚本用于评估网球检测算法的性能，包括精确率、召回率和F1分数。
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 添加项目根目录到系统路径，确保可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入处理函数和标准结果
from src.process import process_img
from data.results.ground_truth import get_ground_truth

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估网球检测算法')
    parser.add_argument('--images_dir', type=str, default='./data/images',
                        help='测试图像目录')
    parser.add_argument('--output_dir', type=str, default='./data/evaluation',
                        help='评估结果输出目录')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化检测结果')
    return parser.parse_args()

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1: 第一个边界框，格式为 {'x': x1, 'y': y1, 'w': w1, 'h': h1}
        box2: 第二个边界框，格式为 {'x': x2, 'y': y2, 'w': w2, 'h': h2}
        
    Returns:
        IoU值，范围为[0, 1]
    """
    # 计算交集区域
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
    y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    
    # 计算交集面积
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集面积
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    return iou

def evaluate_detection(predictions, ground_truth, iou_threshold=0.5):
    """
    评估检测结果
    
    Args:
        predictions: 预测的检测结果列表
        ground_truth: 标准检测结果列表
        iou_threshold: IoU阈值，超过此阈值认为是正确检测
        
    Returns:
        评估指标字典，包含真阳性、假阳性、假阴性、精确率、召回率和F1分数
    """
    # 初始化评估指标
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # 标记已匹配的标准检测框
    matched_gt = [False] * len(ground_truth)
    
    # 遍历预测结果
    for pred in predictions:
        # 查找最佳匹配的标准检测框
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if matched_gt[i]:
                continue
                
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # 如果找到匹配且IoU超过阈值，则为真阳性
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            true_positives += 1
            matched_gt[best_gt_idx] = True
        else:
            false_positives += 1
    
    # 未匹配的标准检测框为假阴性
    false_negatives = sum(1 for m in matched_gt if not m)
    
    # 计算精确率、召回率和F1分数
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def visualize_detection(img_path, predictions, ground_truth, output_dir):
    """
    可视化检测结果
    
    Args:
        img_path: 图像路径
        predictions: 预测的检测结果列表
        ground_truth: 标准检测结果列表
        output_dir: 输出目录
    """
    # 读取图像
    img = cv2.imread(img_path)
    
    # 绘制标准检测结果（绿色）
    for gt in ground_truth:
        x, y, w, h = gt['x'], gt['y'], gt['w'], gt['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 绘制预测结果（红色）
    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['w'], pred['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # 保存可视化结果
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_vis.jpg")
    cv2.imwrite(output_path, img)
    
    # 保存预测结果
    result_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_result.json")
    with open(result_path, 'w') as f:
        json.dump(predictions, f, indent=2)

def plot_performance_metrics(metrics_by_image, output_dir):
    """
    绘制性能指标图表
    
    Args:
        metrics_by_image: 每个图像的评估指标
        output_dir: 输出目录
    """
    # 提取指标
    image_names = list(metrics_by_image.keys())
    precisions = [metrics_by_image[name]['precision'] for name in image_names]
    recalls = [metrics_by_image[name]['recall'] for name in image_names]
    f1_scores = [metrics_by_image[name]['f1_score'] for name in image_names]
    
    # 绘制性能指标图表
    plt.figure(figsize=(12, 6))
    x = np.arange(len(image_names))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='精确率')
    plt.bar(x, recalls, width, label='召回率')
    plt.bar(x + width, f1_scores, width, label='F1分数')
    
    plt.xlabel('图像')
    plt.ylabel('分数')
    plt.title('每个图像的性能指标')
    plt.xticks(x, [os.path.splitext(name)[0] for name in image_names], rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'performance_by_image.png'))
    
    # 绘制结果分布图表
    plt.figure(figsize=(10, 6))
    
    total_tp = sum(metrics_by_image[name]['true_positives'] for name in image_names)
    total_fp = sum(metrics_by_image[name]['false_positives'] for name in image_names)
    total_fn = sum(metrics_by_image[name]['false_negatives'] for name in image_names)
    
    plt.pie([total_tp, total_fp, total_fn], 
            labels=['真阳性', '假阳性', '假阴性'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'orange'])
    
    plt.title('检测结果分布')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'results_distribution.png'))
    
    # 绘制处理时间图表
    plt.figure(figsize=(12, 6))
    
    processing_times = [metrics_by_image[name]['processing_time'] for name in image_names]
    
    plt.bar(x, processing_times)
    plt.xlabel('图像')
    plt.ylabel('处理时间 (ms)')
    plt.title('每个图像的处理时间')
    plt.xticks(x, [os.path.splitext(name)[0] for name in image_names], rotation=90)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'processing_time.png'))

def run_evaluation(images_dir, output_dir, visualize=False):
    """
    运行评估
    
    Args:
        images_dir: 测试图像目录
        output_dir: 评估结果输出目录
        visualize: 是否可视化检测结果
        
    Returns:
        总体评估指标
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"警告: 目录 {images_dir} 中未找到图像文件")
        return None
    
    # 初始化总体评估指标
    total_metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'total_time': 0.0
    }
    
    # 记录每个图像的评估指标
    metrics_by_image = {}
    
    # 处理每个图像
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        print(f"处理图像: {img_path}")
        
        # 获取标准检测结果
        ground_truth = get_ground_truth(img_file)
        
        # 运行检测算法并计时
        start_time = time.time()
        result_json = process_img(img_path)
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 解析检测结果
        predictions = json.loads(result_json)
        
        # 评估检测结果
        metrics = evaluate_detection(predictions, ground_truth)
        metrics['processing_time'] = processing_time
        
        # 更新总体指标
        total_metrics['true_positives'] += metrics['true_positives']
        total_metrics['false_positives'] += metrics['false_positives']
        total_metrics['false_negatives'] += metrics['false_negatives']
        total_metrics['total_time'] += processing_time
        
        # 记录当前图像的评估指标
        metrics_by_image[img_file] = metrics
        
        # 可视化检测结果
        if visualize:
            visualize_detection(img_path, predictions, ground_truth, output_dir)
    
    # 计算总体精确率、召回率和F1分数
    total_tp = total_metrics['true_positives']
    total_fp = total_metrics['false_positives']
    total_fn = total_metrics['false_negatives']
    
    total_metrics['precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_metrics['recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_metrics['f1_score'] = 2 * total_metrics['precision'] * total_metrics['recall'] / (total_metrics['precision'] + total_metrics['recall']) if (total_metrics['precision'] + total_metrics['recall']) > 0 else 0.0
    total_metrics['avg_time'] = total_metrics['total_time'] / len(image_files)
    
    # 绘制性能指标图表
    plot_performance_metrics(metrics_by_image, output_dir)
    
    # 输出评估结果
    print("\n评估结果:")
    print(f"总图像数: {len(image_files)}")
    print(f"真阳性 (TP): {total_tp}")
    print(f"假阳性 (FP): {total_fp}")
    print(f"假阴性 (FN): {total_fn}")
    print(f"精确率: {total_metrics['precision']:.4f}")
    print(f"召回率: {total_metrics['recall']:.4f}")
    print(f"F1分数: {total_metrics['f1_score']:.4f}")
    print(f"平均处理时间: {total_metrics['avg_time']:.2f} ms")
    
    return total_metrics

def main():
    """主函数"""
    args = parse_args()
    
    # 运行评估
    run_evaluation(args.images_dir, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()