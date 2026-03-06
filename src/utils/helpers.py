#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
包含项目中常用的辅助函数和工具类
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import csv
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger("word_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    安全地加载JSON文件
    
    Args:
        filepath: JSON文件路径
        
    Returns:
        解析后的字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {e}")


def save_json_file(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    安全地保存JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 保存路径
        indent: 缩进空格数
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"保存文件时发生错误: {e}")


def load_csv_file(filepath: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    加载CSV文件
    
    Args:
        filepath: CSV文件路径
        encoding: 文件编码
        
    Returns:
        包含字典的列表
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {filepath}")
    except Exception as e:
        raise RuntimeError(f"读取CSV文件时发生错误: {e}")


def save_csv_file(data: List[Dict[str, Any]], filepath: str, 
                  fieldnames: Optional[List[str]] = None) -> None:
    """
    保存CSV文件
    
    Args:
        data: 要保存的数据列表
        filepath: 保存路径
        fieldnames: 字段名列表
    """
    try:
        if not data:
            raise ValueError("数据为空")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except Exception as e:
        raise RuntimeError(f"保存CSV文件时发生错误: {e}")


def validate_text_list(texts: Union[str, List[str]]) -> List[str]:
    """
    验证并标准化文本输入
    
    Args:
        texts: 单个文本字符串或文本列表
        
    Returns:
        标准化的文本列表
    """
    if isinstance(texts, str):
        return [texts]
    elif isinstance(texts, list):
        # 过滤掉空字符串和非字符串元素
        return [str(text).strip() for text in texts if text and str(text).strip()]
    else:
        raise TypeError("输入必须是字符串或字符串列表")


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    格式化时间戳
    
    Args:
        timestamp: 时间戳，如果为None则使用当前时间
        
    Returns:
        格式化的时间字符串
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        包含各种指标的字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    try:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    except Exception as e:
        raise ValueError(f"计算指标时发生错误: {e}")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1):
        """更新进度"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """完成进度跟踪"""
        self.logger.info(f"{self.description} 完成!")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法运算
    
    Args:
        a: 被除数
        b: 除数
        default: 除数为0时的默认值
        
    Returns:
        除法结果
    """
    return a / b if b != 0 else default


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    
    Args:
        lst: 要分块的列表
        chunk_size: 块大小
        
    Returns:
        分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]