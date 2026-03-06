#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
提供配置的高级管理和验证功能
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .config import Config


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化配置管理器
        
        Args:
            config: 配置实例，如果为None则创建新的实例
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self._validators = {}
        self._setup_validators()
    
    def _setup_validators(self) -> None:
        """设置配置验证器"""
        self._validators = {
            'api.port': self._validate_port,
            'training.test_size': self._validate_test_size,
            'detection.threshold': self._validate_threshold,
        }
    
    def _validate_port(self, value: int) -> bool:
        """验证端口号"""
        return isinstance(value, int) and 1 <= value <= 65535
    
    def _validate_test_size(self, value: float) -> bool:
        """验证测试集比例"""
        return isinstance(value, (int, float)) and 0 < value < 1
    
    def _validate_threshold(self, value: float) -> bool:
        """验证阈值"""
        return isinstance(value, (int, float)) and 0 <= value <= 1
    
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误信息列表
        """
        errors = []
        
        for key_path, validator in self._validators.items():
            value = self.config.get(key_path)
            if value is not None and not validator(value):
                errors.append(f"配置项 '{key_path}' 的值 '{value}' 无效")
        
        # 检查必要文件路径
        required_paths = [
            'model.embedding_model_path',
            'training.data_path'
        ]
        
        for path_key in required_paths:
            path_value = self.config.get(path_key)
            if path_value:
                path = Path(path_value)
                if not path.exists():
                    errors.append(f"配置项 '{path_key}' 指定的路径 '{path_value}' 不存在")
        
        return errors
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型相关配置"""
        return self.config.get('model', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取API相关配置"""
        return self.config.get('api', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练相关配置"""
        return self.config.get('training', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """获取检测相关配置"""
        return self.config.get('detection', {})
    
    def update_from_env(self) -> None:
        """从环境变量更新配置"""
        import os
        
        env_mappings = {
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'MODEL_PATH': 'model.embedding_model_path',
            'DATA_PATH': 'training.data_path',
            'LOG_LEVEL': 'logging.level',
        }
        
        updates = {}
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 尝试转换类型
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                
                updates[config_key] = value
        
        if updates:
            self.config.update(updates)
            self.logger.info(f"从环境变量更新了 {len(updates)} 个配置项")
    
    def export_config(self, filepath: str) -> None:
        """
        导出配置到文件
        
        Args:
            filepath: 导出文件路径
        """
        try:
            export_path = Path(filepath)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.config.all_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置已导出到: {filepath}")
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
            raise
    
    def reset_to_default(self) -> None:
        """重置为默认配置"""
        self.config._config = self.config.get_default_config()
        self.config.save_config()
        self.logger.info("配置已重置为默认值")


# 全局配置管理器实例
config_manager = ConfigManager()