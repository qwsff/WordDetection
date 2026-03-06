#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
统一管理项目的所有配置参数
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.json
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "src" / "config" / "config.json"
        
        self.config_path = Path(config_path)
        self._config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                # 如果配置文件不存在，使用默认配置
                self._config = self.get_default_config()
                self.save_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self._config = self.get_default_config()
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 模型配置
            "model": {
                "embedding_model_path": "models/bge-small-zh-v1.5",
                "device": "cpu",
                "max_length": 512
            },
            
            # 数据配置
            "data": {
                "train_file": "data/labeled_multiclass_data.csv",
                "encoding": "utf-8"
            },
            
            # 训练配置
            "training": {
                "seed": 42,
                "test_size": 0.2,
                "stratify": True
            },
            
            # 分类器配置
            "classifier": {
                "type": "logistic_regression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs"
            },
            
            # 模型输出配置
            "output": {
                "classifier_path": "models/classifiers/forbidden_multiclass_classifier.pkl",
                "label_encoder_path": "models/encoders/label_encoder.pkl",
                "logs_dir": "logs"
            },
            
            # API配置
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
                "workers": 1
            },
            
            # 检测配置
            "detection": {
                "threshold": 0.5,
                "return_probability": True,
                "return_embedding": False
            },
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/detection.log"
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，如 'model.embedding_model_path'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self._config
        
        # 导航到倒数第二层
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最后一层的值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        批量更新配置
        
        Args:
            updates: 更新的配置字典
        """
        def _update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update_dict(d[k], v)
                else:
                    d[k] = v
        
        _update_dict(self._config, updates)
        self.save_config()
    
    @property
    def all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()


# 全局配置实例
config = Config()