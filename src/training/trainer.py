#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
违规检测模型训练器
基于标注数据训练二分类违规检测模型
"""

import logging
from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers 未安装")

from ..config.manager import config_manager


class BinaryTrainer:
    """二分类训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config or config_manager.get_training_config()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.classifier = None
        self.embeddings = None
        self.labels = None
        
        # 配置参数
        self.data_path = self.config.get('data_path', 'data/labeled_data.csv')
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.model_path = self.config.get('embedding_model_path', 'models/bge-small-zh-v1.5')
        self.save_path = self.config.get('classifier_save_path', 'models/classifiers/forbidden_classifier.pkl')
    
    def load_data(self) -> bool:
        """
        加载训练数据
        
        Returns:
            是否加载成功
        """
        try:
            data_path = Path(self.data_path)
            if not data_path.exists():
                self.logger.error(f"数据文件不存在: {data_path}")
                return False
            
            # 读取CSV数据
            df = pd.read_csv(data_path, encoding='utf-8')
            
            # 检查必要的列
            required_columns = ['text', 'label']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"数据文件缺少必要列: {required_columns}")
                return False
            
            self.texts = df['text'].tolist()
            self.labels = df['label'].tolist()
            
            self.logger.info(f"成功加载 {len(self.texts)} 条训练数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return False
    
    def encode_texts(self) -> bool:
        """
        对文本进行编码
        
        Returns:
            是否编码成功
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("sentence-transformers 未安装")
            return False
        
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.logger.error(f"嵌入模型路径不存在: {model_path}")
                return False
            
            self.logger.info(f"正在加载嵌入模型: {model_path}")
            
            # 尝试不同的加载方式
            try:
                # 方式1: 直接加载本地模型
                self.model = SentenceTransformer(str(model_path))
            except Exception as e1:
                self.logger.warning(f"直接加载失败: {e1}")
                try:
                    # 方式2: 从HF hub加载同名模型
                    model_name = "BAAI/bge-small-zh-v1.5"
                    self.logger.info(f"尝试从HF加载: {model_name}")
                    self.model = SentenceTransformer(model_name)
                except Exception as e2:
                    self.logger.error(f"HF加载也失败: {e2}")
                    return False
            
            self.logger.info("正在对文本进行编码...")
            self.embeddings = self.model.encode(self.texts)
            self.logger.info(f"编码完成，向量维度: {self.embeddings.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"文本编码失败: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        训练分类模型
        
        Returns:
            是否训练成功
        """
        try:
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                self.embeddings, self.labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.labels
            )
            
            self.logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            
            # 训练逻辑回归分类器
            self.logger.info("正在训练逻辑回归分类器...")
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            self.classifier.fit(X_train, y_train)
            
            # 评估模型
            train_score = self.classifier.score(X_train, y_train)
            test_score = self.classifier.score(X_test, y_test)
            
            self.logger.info(f"训练集准确率: {train_score:.4f}")
            self.logger.info(f"测试集准确率: {test_score:.4f}")
            
            # 详细评估报告
            y_pred = self.classifier.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=['正常', '违规'])
            self.logger.info(f"分类报告:\n{report}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return False
    
    def save_model(self) -> bool:
        """
        保存训练好的模型
        
        Returns:
            是否保存成功
        """
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.classifier, save_path)
            self.logger.info(f"模型已保存到: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """
        完整的训练流程
        
        Returns:
            训练结果字典
        """
        result = {
            "success": False,
            "message": "",
            "metrics": {}
        }
        
        try:
            # 步骤1: 加载数据
            if not self.load_data():
                result["message"] = "数据加载失败"
                return result
            
            # 步骤2: 编码文本
            if not self.encode_texts():
                result["message"] = "文本编码失败"
                return result
            
            # 步骤3: 训练模型
            if not self.train_model():
                result["message"] = "模型训练失败"
                return result
            
            # 步骤4: 保存模型
            if not self.save_model():
                result["message"] = "模型保存失败"
                return result
            
            result["success"] = True
            result["message"] = "训练完成"
            result["metrics"] = {
                "total_samples": len(self.texts),
                "feature_dimension": self.embeddings.shape[1],
                "train_samples": int(len(self.texts) * (1 - self.test_size)),
                "test_samples": int(len(self.texts) * self.test_size)
            }
            
            self.logger.info("训练流程完成")
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            result["message"] = str(e)
        
        return result


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='违规内容检测模型训练器')
    parser.add_argument('--data-path', type=str, help='训练数据路径')
    parser.add_argument('--model-path', type=str, help='嵌入模型路径')
    parser.add_argument('--save-path', type=str, help='模型保存路径')
    parser.add_argument('--test-size', type=float, help='测试集比例')
    
    args = parser.parse_args()
    
    # 更新配置
    config_updates = {}
    if args.data_path:
        config_updates['training.data_path'] = args.data_path
    if args.model_path:
        config_updates['model.embedding_model_path'] = args.model_path
    if args.save_path:
        config_updates['model.classifier_save_path'] = args.save_path
    if args.test_size:
        config_updates['training.test_size'] = args.test_size
    
    if config_updates:
        config_manager.config.update(config_updates)
    
    # 创建训练器并开始训练
    trainer = BinaryTrainer()
    result = trainer.train()
    
    if result["success"]:
        print("✅ 训练成功!")
        print(f"📊 训练结果: {result['metrics']}")
    else:
        print("❌ 训练失败!")
        print(f"📝 错误信息: {result['message']}")

if __name__ == "__main__":
    main()