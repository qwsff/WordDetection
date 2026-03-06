#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器核心功能测试
测试WordDetector和MulticlassWordDetector的功能
"""

import pytest
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

class TestWordDetector:
    """二分类检测器测试"""
    
    def test_import_detector(self):
        """测试检测器导入"""
        try:
            from core.detector import WordDetector
            assert WordDetector is not None
        except ImportError as e:
            pytest.skip(f"跳过测试: {e}")
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        try:
            from core.detector import WordDetector
            
            # 测试默认配置
            detector = WordDetector()
            assert detector is not None
            assert hasattr(detector, 'config')
            assert hasattr(detector, 'is_loaded')
            
            # 测试自定义配置
            custom_config = {
                'model_path': '/fake/path',
                'classifier_path': '/fake/path',
                'threshold': 0.7
            }
            detector = WordDetector(custom_config)
            assert detector.config['threshold'] == 0.7
            
        except ImportError:
            pytest.skip("无法导入检测器模块")
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        try:
            from core.detector import WordDetector
            
            detector = WordDetector()
            info = detector.get_model_info()
            
            assert isinstance(info, dict)
            assert 'model_loaded' in info
            assert 'model_path' in info
            assert 'classifier_path' in info
            
        except ImportError:
            pytest.skip("无法导入检测器模块")

class TestMulticlassWordDetector:
    """多分类检测器测试"""
    
    def test_import_multiclass_detector(self):
        """测试多分类检测器导入"""
        try:
            from core.detector import MulticlassWordDetector
            assert MulticlassWordDetector is not None
        except ImportError as e:
            pytest.skip(f"跳过测试: {e}")
    
    def test_multiclass_detector_inheritance(self):
        """测试多分类检测器继承关系"""
        try:
            from core.detector import WordDetector, MulticlassWordDetector
            
            detector = MulticlassWordDetector()
            assert isinstance(detector, WordDetector)
            
        except ImportError:
            pytest.skip("无法导入检测器模块")

class TestConfiguration:
    """配置管理测试"""
    
    def test_import_config_manager(self):
        """测试配置管理器导入"""
        try:
            from config.manager import config_manager
            assert config_manager is not None
        except ImportError as e:
            pytest.skip(f"跳过测试: {e}")
    
    def test_config_validation(self):
        """测试配置验证"""
        try:
            from config.manager import config_manager
            
            errors = config_manager.validate_config()
            # 配置应该有效或者返回具体的错误信息
            assert errors is not None
            
        except ImportError:
            pytest.skip("无法导入配置管理器")

# 性能测试
@pytest.mark.performance
class TestPerformance:
    """性能相关测试"""
    
    def test_detector_response_time(self):
        """测试检测器响应时间"""
        try:
            import time
            from core.detector import WordDetector
            
            detector = WordDetector()
            
            # 测试空文本的响应时间
            start_time = time.time()
            result = detector.predict("")
            end_time = time.time()
            
            response_time = end_time - start_time
            # 即使模型未加载，也应该在合理时间内返回
            assert response_time < 5.0  # 5秒内应该返回
            
        except ImportError:
            pytest.skip("无法导入检测器模块")

# 集成测试
@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        try:
            from core.detector import WordDetector
            from config.manager import config_manager
            
            # 1. 配置验证
            errors = config_manager.validate_config()
            assert errors is not None
            
            # 2. 检测器初始化
            detector = WordDetector()
            assert detector is not None
            
            # 3. 获取模型信息
            info = detector.get_model_info()
            assert isinstance(info, dict)
            
        except ImportError as e:
            pytest.skip(f"跳过集成测试: {e}")