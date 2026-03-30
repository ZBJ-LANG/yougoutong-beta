"""
子模块基础框架 - 定义统一的接口规范
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import pandas as pd


class BaseRecommendationModule(ABC):
    """
    推荐子模块的抽象基类，定义统一的接口规范
    """
    
    def __init__(self, module_name: str, category: str):
        """
        初始化子模块
        
        Args:
            module_name: 模块名称，如"电子数码"
            category: 商品类别，如"电子产品"
        """
        self.module_name = module_name
        self.category = category
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载训练数据
        
        Args:
            data_path: 数据文件路径
        
        Returns:
            pd.DataFrame: 加载后的数据
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        pass
    
    @abstractmethod
    def feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        特征工程
        
        Args:
            data: 预处理后的数据
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 特征工程后的数据和特征元数据
        """
        pass
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            **kwargs: 训练参数
        """
        pass
    
    @abstractmethod
    def predict(self, user_features: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        生成推荐结果
        
        Args:
            user_features: 用户特征
            top_k: 返回推荐结果的数量
        
        Returns:
            List[Dict[str, Any]]: 推荐结果列表，每个元素包含商品信息和推荐分数
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        离线评估模型
        
        Args:
            test_data: 测试数据
        
        Returns:
            Dict[str, float]: 评估指标，如准确率、召回率、NDCG等
        """
        pass
    
    def save_model(self, model_path: str) -> None:
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, model_path: str) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        import pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def get_module_info(self) -> Dict[str, Any]:
        """
        获取模块信息
        
        Returns:
            Dict[str, Any]: 模块信息
        """
        return {
            "module_name": self.module_name,
            "category": self.category,
            "is_trained": self.is_trained
        }
