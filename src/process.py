#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能捡网球机器人目标检测算法 - 检测实现

本模块实现网球检测算法，包括图像预处理、特征提取和目标检测。
"""

import os
import time
import cv2
import numpy as np
import json

# 模型相关导入
try:
    from mindspore import context, Tensor
    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    import mindspore.ops as ops
    from mindspore.common import dtype as mstype
    HAS_MINDSPORE = True
except ImportError:
    HAS_MINDSPORE = False
    print("警告: MindSpore未安装，将使用OpenCV进行基础检测")

# 全局变量
MODEL_INITIALIZED = False
MODEL = None
DEVICE_ID = 0
INPUT_SIZE = 640  # 模型输入尺寸
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.45  # NMS IOU阈值

def initialize_model():
    """
    初始化模型，加载预训练权重
    """
    global MODEL_INITIALIZED, MODEL
    
    if MODEL_INITIALIZED:
        return True
    
    try:
        if HAS_MINDSPORE:
            # 设置MindSpore上下文
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
            
            # 这里应该导入自定义的模型类并初始化
            # 示例: from model import YOLOModel
            # MODEL = YOLOModel()
            
            # 加载预训练权重
            # param_dict = load_checkpoint("path_to_checkpoint.ckpt")
            # load_param_into_net(MODEL, param_dict)
            
            # 由于没有实际模型，这里仅作为示例
            print("模型已初始化（示例）")
            MODEL_INITIALIZED = True
            return True
        else:
            print("MindSpore未安装，将使用OpenCV进行基础检测")
            MODEL_INITIALIZED = True
            return True
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return False

def preprocess_image(img):
    """
    图像预处理
    
    Args:
        img: 输入图像
        
    Returns:
        预处理后的图像，原始图像尺寸
    """
    original_shape = img.shape[:2]  # (height, width)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    
    # 归一化
    normalized_img = resized_img / 255.0
    
    if HAS_MINDSPORE:
        # 转换为MindSpore Tensor，并调整维度顺序为NCHW
        input_tensor = np.transpose(normalized_img, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加batch维度
        input_tensor = Tensor(input_tensor, mstype.float32)
        return input_tensor, original_shape
    else:
        return normalized_img, original_shape

def detect_tennis_balls_opencv(img):
    """
    使用OpenCV的增强方法检测网球
    
    Args:
        img: 输入图像
        
    Returns:
        检测到的网球列表，每个网球包含x, y, w, h信息
    """
    # 创建原始图像的副本
    img_copy = img.copy()
    
    # 应用高斯模糊减少噪声
    img_blur = cv2.GaussianBlur(img_copy, (5, 5), 0)
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    # 应用CLAHE（对比度受限的自适应直方图均衡化）增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # 转换为列表，使其可修改
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 定义更严格的颜色范围以减少误检
    # 黄绿色范围（标准网球颜色）- 更严格的范围
    lower_yellow1 = np.array([25, 120, 120])
    upper_yellow1 = np.array([40, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
    
    # 应用形态学操作改进掩码质量
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 开运算（先腐蚀后膨胀）去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # 闭运算（先膨胀后腐蚀）填充目标内小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 使用Hough圆检测辅助识别
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,  # 增加分辨率
        minDist=30,  # 增加最小距离
        param1=100,  # 增加边缘检测阈值
        param2=40,   # 增加圆心检测阈值
        minRadius=10,  # 设置最小半径
        maxRadius=60   # 限制最大半径
    )
    
    # 存储所有可能的网球检测结果
    potential_balls = []
    
    # 处理轮廓检测结果
    for contour in contours:
        # 计算轮廓面积，过滤小区域
        area = cv2.contourArea(contour)
        if area < 30:  # 增加最小面积阈值
            continue
            
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤过小的检测框
        if w < 10 or h < 10:
            continue
            
        # 计算长宽比，网球应该接近正方形
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # 更严格的长宽比
            continue
            
        # 计算圆度，网球应该接近圆形
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 计算填充率（轮廓面积与边界框面积之比）
        fill_ratio = area / (w * h) if (w * h) > 0 else 0
        
        # 根据特征计算置信度分数
        confidence = (circularity * 0.6 + fill_ratio * 0.4) if circularity > 0.5 else 0
        
        # 更严格的圆度和填充率条件
        if circularity > 0.5 and fill_ratio > 0.5 and confidence > 0.6:
            potential_balls.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": confidence,
                "source": "contour"
            })
    
    # 处理Hough圆检测结果
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # 圆心和半径
            center_x, center_y, radius = circle
            
            # 过滤半径过小的圆
            if radius < 10:
                continue
                
            # 转换为边界框格式
            x = max(0, int(center_x - radius))
            y = max(0, int(center_y - radius))
            w = int(radius * 2)
            h = int(radius * 2)
            
            # 确保坐标在图像范围内
            if x + w > img.shape[1]:
                w = img.shape[1] - x
            if y + h > img.shape[0]:
                h = img.shape[0] - y
                
            # 检查区域颜色是否符合网球特征
            roi = hsv[y:y+h, x:x+w]
            if roi.size == 0:  # 确保ROI不为空
                continue
                
            # 计算区域内黄色像素的比例
            yellow_mask = cv2.inRange(roi, lower_yellow1, upper_yellow1)
            yellow_ratio = np.sum(yellow_mask > 0) / (w * h) if (w * h) > 0 else 0
            
            # 只有当黄色像素比例足够高时才认为是网球
            if yellow_ratio > 0.3:
                potential_balls.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": 0.7 * yellow_ratio,  # 根据黄色比例调整置信度
                    "source": "hough"
                })
    
    # 应用非极大值抑制(NMS)去除重叠检测
    results = non_max_suppression(potential_balls, iou_threshold=0.3)
    
    # 按面积从大到小排序
    results.sort(key=lambda x: x["w"] * x["h"], reverse=True)
    
    # 只保留置信度较高的结果
    filtered_results = [r for r in results if r["confidence"] > 0.6]
    
    # 如果没有高置信度的结果，尝试检测小球
    if not filtered_results:
        # 针对小球的特殊处理
        small_balls = detect_small_tennis_balls(img)
        if small_balls:
            filtered_results = small_balls
    
    return filtered_results

def detect_small_tennis_balls(img):
    """
    专门用于检测小尺寸网球的函数
    
    Args:
        img: 输入图像
        
    Returns:
        检测到的小网球列表
    """
    # 创建原始图像的副本
    img_copy = img.copy()
    
    # 应用高斯模糊减少噪声
    img_blur = cv2.GaussianBlur(img_copy, (3, 3), 0)
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    # 定义小球的颜色范围（可能更亮或更暗）
    lower_yellow_small = np.array([20, 80, 80])
    upper_yellow_small = np.array([45, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_yellow_small, upper_yellow_small)
    
    # 应用形态学操作
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    small_balls = []
    
    # 处理轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 只考虑面积在特定范围内的轮廓
        if 10 < area < 200:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算圆度
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 如果圆度足够高，认为是小球
            if circularity > 0.4:
                small_balls.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": circularity,
                    "source": "small"
                })
    
    # 应用非极大值抑制
    results = non_max_suppression(small_balls, iou_threshold=0.3)
    
    # 按面积从大到小排序
    results.sort(key=lambda x: x["w"] * x["h"], reverse=True)
    
    return results

def non_max_suppression(boxes, iou_threshold=0.3):
    """
    非极大值抑制，去除重叠的检测框
    
    Args:
        boxes: 检测框列表，每个元素为包含x, y, w, h, confidence的字典
        iou_threshold: IoU阈值，超过此阈值的框将被抑制
        
    Returns:
        保留的检测框列表
    """
    if not boxes:
        return []
    
    # 按置信度排序
    boxes.sort(key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    
    while boxes:
        # 取置信度最高的框
        current = boxes.pop(0)
        keep.append(current)
        
        # 计算当前框与剩余框的IoU
        i = 0
        while i < len(boxes):
            iou = calculate_iou(current, boxes[i])
            if iou > iou_threshold:
                boxes.pop(i)
            else:
                i += 1
    
    return keep

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

def postprocess_results(detections, original_shape):
    """
    后处理检测结果
    
    Args:
        detections: 模型输出的检测结果
        original_shape: 原始图像尺寸
        
    Returns:
        处理后的检测结果列表
    """
    processed_results = []
    
    # 将检测结果转换为所需格式并按面积排序
    for det in detections:
        # 只保留必要的字段
        processed_results.append({
            "x": det["x"],
            "y": det["y"],
            "w": det["w"],
            "h": det["h"]
        })
    
    # 按面积从大到小排序
    processed_results.sort(key=lambda x: x["w"] * x["h"], reverse=True)
    
    return processed_results

#
# 参数:
#   img_path: 要识别的图片的路径
#
# 返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 "图片对应输出结果.txt" 中一张图片对应的结果
#
def process_img(img_path):
    """
    处理图像并识别网球
    
    Args:
        img_path: 要识别的图片的路径
        
    Returns:
        JSON格式的识别结果，包含网球的位置信息(x, y, w, h)，按面积从大到小排序
    """
    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"错误: 文件 {img_path} 不存在")
        return json.dumps([])
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像 {img_path}")
        return json.dumps([])
    
    # 初始化模型（如果尚未初始化）
    if not MODEL_INITIALIZED:
        initialize_model()
    
    if HAS_MINDSPORE and MODEL is not None:
        # 预处理图像
        input_tensor, original_shape = preprocess_image(img)
        
        # 模型推理
        # 这里应该调用实际模型进行推理
        # 示例: output = MODEL(input_tensor)
        
        # 由于没有实际模型，这里使用OpenCV方法作为替代
        detections = detect_tennis_balls_opencv(img)
        
        # 后处理结果
        results = postprocess_results(detections, original_shape)
    else:
        # 使用OpenCV方法进行基础检测
        results = detect_tennis_balls_opencv(img)
        results = postprocess_results(results, img.shape[:2])
    
    # 返回JSON格式的结果
    return json.dumps(results)

#
# 以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
# 但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
# 因此提交时请根据情况删除不必要的额外代码
#
if __name__=='__main__':
    imgs_folder = './imgs/'
    img_paths = os.listdir(imgs_folder)
    def now():
        return int(time.time()*1000)
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    for img_path in img_paths:
        print(img_path,':')
        last_time = now()
        result = process_img(imgs_folder+img_path)
        run_time = now() - last_time
        print('result:\n',result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    print('\n')
    print('avg time: ',int(count_time/len(img_paths)),'ms')
    print('max time: ',max_time,'ms')
    print('min time: ',min_time,'ms')