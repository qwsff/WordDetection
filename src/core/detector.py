#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心检测器模块
封装违规内容检测的核心逻辑
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers 未安装，部分功能可能不可用")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("joblib 未安装，模型加载功能不可用")


class WordDetector:
    """违规内容检测器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.classifier = None
        self.is_loaded = False
        
        # 配置默认值
        self.model_path = self.config.get('model_path', 'models/bge-small-zh-v1.5')
        self.classifier_path = self.config.get('classifier_path', 'models/classifiers/forbidden_classifier.pkl')
        self.threshold = self.config.get('threshold', 0.5)
        self.offline_mode = self.config.get('offline_mode', False)
        
    def load_models(self) -> bool:
        """
        加载模型和分类器
        
        Returns:
            是否加载成功
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not JOBLIB_AVAILABLE:
            self.logger.error("缺少必要的依赖包")
            return False
            
        try:
            # 检查离线模式设置
            if self.offline_mode or os.getenv('SENTENCE_TRANSFORMERS_OFFLINE') == '1':
                self.logger.info("🔄 运行在离线模式，仅使用本地模型")
                os.environ['SENTENCE_TRANSFORMERS_OFFLINE'] = '1'
            
            # 加载嵌入模型
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.logger.error(f"嵌入模型路径不存在: {model_path}")
                if self.offline_mode:
                    self.logger.error("离线模式下无法从网络下载模型")
                    return False
                return self._load_from_hf_fallback()
                
            self.logger.info(f"正在加载嵌入模型: {model_path}")
            
            # 优先尝试本地加载
            if self._load_local_model(model_path):
                self.logger.info("✅ 本地模型加载成功")
            elif not self.offline_mode:
                # 如果本地加载失败且非离线模式，尝试从HF加载
                self.logger.warning("本地模型加载失败，尝试从HF加载")
                if not self._load_from_hf():
                    return False
            else:
                self.logger.error("本地模型加载失败且处于离线模式")
                return False
            
            # 加载分类器
            classifier_path = Path(self.classifier_path)
            if not classifier_path.exists():
                self.logger.error(f"分类器模型路径不存在: {classifier_path}")
                return False
                
            self.logger.info(f"正在加载分类器: {classifier_path}")
            self.classifier = joblib.load(classifier_path)
            
            self.is_loaded = True
            self.logger.info("🎯 模型加载完成")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False
    
    def _load_local_model(self, model_path: Path) -> bool:
        """尝试加载本地模型"""
        try:
            # 检查必要的模型文件
            required_files = ['config.json', 'modules.json', 'pytorch_model.bin']
            for file in required_files:
                if not (model_path / file).exists():
                    self.logger.debug(f"缺少必要文件: {file}")
                    return False
            
            self.model = SentenceTransformer(str(model_path))
            return True
        except Exception as e:
            self.logger.debug(f"本地模型加载失败: {e}")
            return False
    
    def _load_from_hf(self) -> bool:
        """从HuggingFace加载模型"""
        try:
            model_name = "BAAI/bge-small-zh-v1.5"
            self.logger.info(f"尝试从HF加载: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # 可选：保存到本地供下次使用
            local_path = Path(self.model_path)
            if not local_path.exists():
                self.logger.info(f"保存模型到本地: {local_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(local_path))
            
            return True
        except Exception as e:
            self.logger.error(f"HF模型加载失败: {e}")
            return False
    
    def _load_from_hf_fallback(self) -> bool:
        """当本地模型不存在时的备用加载方案"""
        if self.offline_mode:
            return False
        
        self.logger.info("本地模型不存在，从HF下载...")
        return self._load_from_hf()
    
    def predict(self, texts: Union[str, List[str]], 
                return_probability: bool = True,
                return_embedding: bool = False) -> Dict[str, Any]:
        """
        预测文本是否违规
        
        Args:
            texts: 待检测的文本或文本列表
            return_probability: 是否返回概率
            return_embedding: 是否返回嵌入向量
            
        Returns:
            检测结果字典
        """
        if not self.is_loaded:
            return {"error": "模型未加载"}
        
        # 统一处理单个文本和文本列表
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        try:
            # 生成嵌入向量
            embeddings = self.model.encode(texts)
            
            # 预测
            predictions = self.classifier.predict(embeddings)
            probabilities = self.classifier.predict_proba(embeddings) if return_probability else None
            
            # 构造结果
            results = []
            for i, text in enumerate(texts):
                result = {
                    "text": text,
                    "is_forbidden": bool(predictions[i]),
                    "prediction": "违规" if predictions[i] else "正常"
                }
                
                if return_probability and probabilities is not None:
                    result["probability"] = float(probabilities[i][1])
                    result["confidence"] = float(max(probabilities[i]))
                
                if return_embedding:
                    result["embedding"] = embeddings[i].tolist()
                
                results.append(result)
            
            # 如果是单个文本，返回单个结果
            if is_single:
                return results[0]
            else:
                return {
                    "results": results,
                    "total_count": len(results),
                    "forbidden_count": sum(1 for r in results if r["is_forbidden"])
                }
                
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {e}")
            return {"error": str(e)}
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        批量预测（适用于大量文本）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            预测结果列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_result = self.predict(batch_texts, return_probability=True)
            if "results" in batch_result:
                results.extend(batch_result["results"])
            else:
                results.append(batch_result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_loaded": self.is_loaded,
            "model_path": self.model_path,
            "classifier_path": self.classifier_path,
            "threshold": self.threshold,
            "offline_mode": self.offline_mode,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else None
        }


class MulticlassWordDetector(WordDetector):
    """多分类违规检测器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.class_labels = []  # 类别标签列表
        
    def load_models(self) -> bool:
        """加载多分类模型"""
        success = super().load_models()
        if success and hasattr(self.classifier, 'classes_'):
            self.class_labels = [str(label) for label in self.classifier.classes_]
        return success
    
    def predict(self, texts: Union[str, List[str]], 
                return_probability: bool = True,
                return_embedding: bool = False) -> Dict[str, Any]:
        """多分类预测"""
        if not self.is_loaded:
            return {"error": "模型未加载"}
        
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        try:
            embeddings = self.model.encode(texts)
            predictions = self.classifier.predict(embeddings)
            probabilities = self.classifier.predict_proba(embeddings) if return_probability else None
            
            results = []
            for i, text in enumerate(texts):
                result = {
                    "text": text,
                    "predicted_class": str(predictions[i]),
                    "class_label": self.class_labels[predictions[i]] if self.class_labels else str(predictions[i])
                }
                
                if return_probability and probabilities is not None:
                    prob_dict = {}
                    for j, label in enumerate(self.class_labels):
                        prob_dict[label] = float(probabilities[i][j])
                    result["probabilities"] = prob_dict
                    result["confidence"] = float(max(probabilities[i]))
                
                if return_embedding:
                    result["embedding"] = embeddings[i].tolist()
                
                results.append(result)
            
            if is_single:
                return results[0]
            else:
                return {
                    "results": results,
                    "total_count": len(results),
                    "class_distribution": self._get_class_distribution(results)
                }
                
        except Exception as e:
            self.logger.error(f"多分类预测过程中发生错误: {e}")
            return {"error": str(e)}
    
    def _get_class_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """获取类别分布统计"""
        distribution = {}
        for result in results:
            label = result["class_label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution