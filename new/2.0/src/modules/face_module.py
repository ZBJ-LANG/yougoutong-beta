#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美妆护肤推荐子模块
"""
from typing import Dict, List, Any, Tuple
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score
from lightgbm import LGBMClassifier

# 添加当前目录到系统路径，确保可以导入base_module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from base_module import BaseRecommendationModule
except ImportError:
    # 如果直接运行，使用抽象基类定义
    from abc import ABC, abstractmethod
    class BaseRecommendationModule(ABC):
        def __init__(self, module_name: str, category: str):
            self.module_name = module_name
            self.category = category
            self.model = None
            self.is_trained = False
        @abstractmethod
        def load_data(self, data_path: str):
            pass
        @abstractmethod
        def preprocess_data(self, data):
            pass
        @abstractmethod
        def feature_engineering(self, data):
            pass
        @abstractmethod
        def train(self, train_data):
            pass
        @abstractmethod
        def predict(self, user_features, top_k=10):
            pass
        @abstractmethod
        def evaluate(self, test_data):
            pass


class FaceRecommendationModule(BaseRecommendationModule):
    """
    美妆护肤推荐子模块，结合LightGBM模型和真实知识图谱实现商品推荐
    """
    
    def __init__(self):
        """
        初始化美妆护肤推荐模块
        """
        super().__init__(module_name="美妆护肤", category="美妆护肤")
        self.feature_meta = {}
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.knowledge_graph = None
        self.fusion_service = None
        self.local_product_data = []  # 存储本地商品数据作为备用
        
        # 尝试加载真实知识图谱
        self._load_knowledge_graph()
        
        # 尝试加载融合服务
        self._load_fusion_service()
        
        # 加载本地商品数据作为备用
        self._load_local_product_data()
    
    def _load_knowledge_graph(self) -> None:
        """
        加载真实知识图谱
        """
        try:
            # 先导入FaceKnowledgeGraph类
            from .face_knowledge_graph import FaceKnowledgeGraph
            # 确保类在当前模块的命名空间中
            import sys
            sys.modules['__main__'].FaceKnowledgeGraph = FaceKnowledgeGraph
            
            # 使用绝对路径来检查文件是否存在
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "face_knowledge_graph.pkl")
            print(f"检查知识图谱文件路径: {model_path}")
            if os.path.exists(model_path):
                self.knowledge_graph = FaceKnowledgeGraph.load_graph(model_path)
                print("✅ 成功加载美妆护肤模块真实知识图谱")
            else:
                print("⚠️  知识图谱文件不存在，将使用LightGBM模型推荐")
        except Exception as e:
            print(f"❌ 加载知识图谱失败: {e}")
            self.knowledge_graph = None
    
    def _load_fusion_service(self) -> None:
        """
        加载向量数据库与知识图谱融合服务
        """
        try:
            import sys
            import os
            # 添加根目录到路径
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from fusion_service import FusionService
            self.fusion_service = FusionService(category="face")
            print("✅ 成功加载美妆护肤模块融合服务")
        except Exception as e:
            print(f"❌ 加载融合服务失败: {e}")
            self.fusion_service = None
            # 如果融合服务加载失败，尝试直接填充向量数据库
            self._populate_vector_db_directly()
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载美妆护肤数据
        
        Args:
            data_path: 数据文件路径
        
        Returns:
            pd.DataFrame: 加载后的数据
        """
        # 加载CSV数据，尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        data = None
        
        for encoding in encodings:
            try:
                data = pd.read_csv(data_path, encoding=encoding)
                print(f"使用编码 {encoding} 加载数据完成，共 {len(data)} 条记录")
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            raise ValueError("无法读取数据文件，请检查文件编码")
            
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        美妆护肤数据预处理
        
        Args:
            data: 原始数据
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 1. 检查数据格式，适配不同的数据结构
        if '用户ID' in data.columns:  # 用户行为数据格式
            # 重命名列，统一数据格式
            data = data.rename(columns={
                '用户ID': 'user_id',
                '商品ID': 'product_id',
                '品牌': 'brand',
                '商品名称': 'product_name',
                '商品类别': 'category',
                '行为类型': 'behavior_type',
                '时间戳': 'event_time',
                '售价': 'price',
                '价格区间': 'price_range'
            })
            
            # 2. 处理缺失值
            data = data.dropna(subset=['user_id', 'product_id', 'price'])
            
            # 3. 转换数据类型
            data['user_id'] = data['user_id'].astype(str)
            data['product_id'] = data['product_id'].astype(str)
            if 'event_time' in data.columns:
                if data['event_time'].dtype == 'object':
                    try:
                        data['event_time'] = pd.to_datetime(data['event_time'])
                    except:
                        # 如果转换失败，将时间戳作为字符串处理
                        data['event_time'] = data['event_time'].astype(str)
            
            # 4. 添加虚拟order_id（如果没有）
            if 'order_id' not in data.columns:
                if 'event_time' in data.columns and data['event_time'].dtype == 'datetime64[ns]':
                    data['order_id'] = data['event_time'].astype(str) + '_' + data['user_id']
                else:
                    data['order_id'] = data['user_id'] + '_' + data['product_id']
        
        elif 'ID' in data.columns:  # 美妆商品信息数据格式（适配实际数据列名）
            # 重命名列，统一数据格式
            data = data.rename(columns={
                'ID': 'product_id',
                '名称': 'product_name',
                '商家': 'brand',
                '单价': 'price',
                '评价': 'rating',
                '品类': 'category'
            })
            
            # 2. 处理缺失值
            data = data.dropna(subset=['product_id', 'price'])
            
            # 3. 转换数据类型
            data['product_id'] = data['product_id'].astype(str)
        
        elif '商品ID' in data.columns:  # 其他商品信息数据格式
            # 重命名列，统一数据格式
            data = data.rename(columns={
                '商品ID': 'product_id',
                '商品名称': 'product_name',
                '品牌': 'brand',
                '品类': 'category',
                '单价': 'price',
                '评价': 'rating',
                '成分': 'ingredients',
                '适用肤质': 'skin_type',
                '功效': 'effect'
            })
            
            # 2. 处理缺失值
            data = data.dropna(subset=['product_id', 'price'])
            
            # 3. 转换数据类型
            data['product_id'] = data['product_id'].astype(str)
        
        # 4. 处理价格列，确保为数值类型
        if 'price' in data.columns:
            if data['price'].dtype == 'object':
                # 移除价格中的非数字字符
                data['price'] = data['price'].str.replace('¥', '').str.replace(',', '').astype(float)
        
        print(f"预处理后数据：{len(data)} 条记录")
        return data
    
    def feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        美妆护肤特征工程
        
        Args:
            data: 预处理后的数据
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 特征工程后的数据和特征元数据
        """
        # 1. 处理类别特征
        if 'brand' in data.columns:
            data['brand'] = data['brand'].fillna('unknown')
        
        if 'category' in data.columns:
            data['category'] = data['category'].fillna('unknown')
        
        if 'skin_type' in data.columns:
            data['skin_type'] = data['skin_type'].fillna('未知')
        
        if 'effect' in data.columns:
            data['effect'] = data['effect'].fillna('未知')
        
        # 2. 计算商品特征
        item_features = pd.DataFrame()
        if 'product_id' in data.columns:
            # 计算商品的基本特征
            item_features = data.groupby('product_id').agg({
                'price': ['mean', 'count'],
                'brand': 'first',
                'category': 'first',
                'rating': lambda x: x.mean() if 'rating' in data.columns and pd.api.types.is_numeric_dtype(x) else x.iloc[0] if len(x) > 0 else 0.0
            }).reset_index()
            
            item_features.columns = ['product_id', 'avg_price', 'sales_count', 'brand', 'category', 'avg_rating']
            item_features['sales_volume'] = item_features['avg_price'] * item_features['sales_count']
            
            # 确保avg_rating为数值类型
            item_features['avg_rating'] = pd.to_numeric(item_features['avg_rating'], errors='coerce').fillna(0.0)
        
        # 3. 计算用户行为特征（基于不同行为类型）
        user_features = pd.DataFrame()
        if 'user_id' in data.columns:
            # 计算不同行为类型的次数
            if 'behavior_type' in data.columns:
                behavior_counts = data.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
                behavior_counts.columns = ['user_id'] + [f'{col}_count' for col in behavior_counts.columns[1:]]
            else:
                behavior_counts = pd.DataFrame(data['user_id'].unique(), columns=['user_id'])
                behavior_counts['purchase_count'] = 1  # 默认购买次数为1
            
            # 计算用户特征
            user_features = data.groupby('user_id').agg({
                'price': ['mean', 'sum', 'std'],
                'brand': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown',
                'category': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
            }).reset_index()
            
            user_features.columns = ['user_id', 'avg_purchase_price', 'total_spent', 'price_std', 
                                    'favorite_brand', 'favorite_category']
            
            # 合并用户行为特征
            user_features = pd.merge(user_features, behavior_counts, on='user_id', how='left')
            
            # 填充缺失的行为计数
            for behavior in ['pv', 'buy', 'cart', 'collect']:
                behavior_col = f'{behavior}_count'
                if behavior_col not in user_features.columns:
                    user_features[behavior_col] = 0
        
        # 4. 构建用户-商品交互矩阵
        interaction_counts = pd.DataFrame()
        if 'user_id' in data.columns and 'product_id' in data.columns:
            if 'behavior_type' in data.columns:
                # 计算用户对商品的不同行为次数
                interaction_counts = data.groupby(['user_id', 'product_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
                
                # 重命名列
                interaction_counts.columns = ['user_id', 'product_id'] + [f'{col}_count' for col in interaction_counts.columns[2:]]
            else:
                # 简化处理，假设所有记录都是购买行为
                interaction_counts = data.groupby(['user_id', 'product_id']).size().reset_index(name='buy_count')
            
            # 计算总交互次数
            interaction_counts['interaction_count'] = interaction_counts.iloc[:, 2:].sum(axis=1)
        
        # 5. 合并所有特征
        final_data = pd.DataFrame()
        
        # 情况1：有完整的用户-商品交互数据
        if not interaction_counts.empty and not user_features.empty and not item_features.empty:
            merged_data = pd.merge(interaction_counts, user_features, on='user_id', how='left')
            merged_data = pd.merge(merged_data, item_features, on='product_id', how='left')
            
            # 生成标签：是否购买过
            if 'buy_count' in merged_data.columns:
                merged_data['label'] = (merged_data['buy_count'] > 0).astype(int)
            else:
                merged_data['label'] = 1
            
            final_data = merged_data
        elif not item_features.empty:
            # 情况2：只有商品数据，生成完整的训练数据
            final_data = item_features.copy()
            final_data['label'] = 1  # 假设所有商品都值得推荐
            
            # 添加必要的特征列，确保模型训练时有足够的特征
            final_data['avg_purchase_price'] = final_data['avg_price'].mean()
            final_data['total_spent'] = final_data['avg_price'] * 3  # 假设平均购买3件
            final_data['price_std'] = final_data['avg_price'].std()
            final_data['favorite_brand'] = final_data['brand']
            final_data['favorite_category'] = final_data['category']
            final_data['purchase_count'] = 1
            final_data['pv_count'] = 5
            final_data['cart_count'] = 2
            final_data['collect_count'] = 1
            final_data['interaction_count'] = 8
        
        # 6. 处理分类特征
        categorical_features = []
        for col in final_data.columns:
            if final_data[col].dtype == 'object' or final_data[col].dtype == 'category':
                categorical_features.append(col)
                final_data[col] = final_data[col].astype('category')
        
        # 7. 保存特征元数据
        numerical_features = [col for col in final_data.columns 
                            if col not in categorical_features and col not in ['user_id', 'product_id', 'label']]
        
        feature_meta = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'all_users': list(data['user_id'].unique()) if 'user_id' in data.columns else [],
            'all_products': list(data['product_id'].unique()) if 'product_id' in data.columns else []
        }
        
        print(f"特征工程完成，最终数据：{len(final_data)} 条记录")
        
        self.feature_meta = feature_meta
        self.user_item_matrix = interaction_counts
        self.item_features = item_features
        self.user_features = user_features
        
        return final_data, feature_meta
    
    def train(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        训练美妆护肤推荐模型
        
        Args:
            train_data: 训练数据
            **kwargs: 训练参数
        """
        if train_data.empty:
            print("⚠️  训练数据为空，跳过训练")
            self.is_trained = False
            return
        
        # 分离特征和标签
        X = train_data.drop(['user_id', 'product_id', 'label'], axis=1, errors='ignore')
        y = train_data['label']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 初始化并训练LightGBM模型
        self.model = LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 5),
            random_state=42
        )
        
        # 训练模型
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc'
        )
        
        self.is_trained = True  # 先设置训练完成标志
        
        # 评估模型
        eval_result = self.evaluate(pd.concat([X_test, y_test], axis=1))
        print(f"模型训练完成，评估结果：{eval_result}")
    
    def predict(self, user_features: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        为用户生成美妆护肤推荐
        
        Args:
            user_features: 用户特征，包含user_id和user_needs
            top_k: 返回推荐结果的数量
        
        Returns:
            List[Dict[str, Any]]: 推荐结果列表
        """
        user_id = user_features.get('user_id')
        user_needs = user_features.get('user_needs', '')
        if not user_id:
            raise ValueError("用户ID不能为空")
        
        # 优先使用融合服务进行推荐
        if self.fusion_service is not None:
            print("🔄 使用融合服务（知识图谱+向量数据库）进行推荐")
            try:
                # 获取融合服务推荐结果
                fusion_recommendations = self.fusion_service.query(user_needs, top_k)
                
                if fusion_recommendations:
                    # 处理融合服务推荐结果，确保格式统一
                    formatted_recommendations = []
                    for rec in fusion_recommendations:
                        formatted_recommendations.append({
                            'product_id': rec['product_id'],
                            'brand': rec.get('name', '').split()[0] if rec.get('name') else '未知',
                            'category': rec.get('name', '').split()[1] if rec.get('name') and len(rec.get('name', '').split()) > 1 else '未知',
                            'price': rec.get('price', 0.0),
                            'sales_count': 0,  # 融合服务中没有销量信息，设为0
                            'score': rec['score']
                        })
                    return formatted_recommendations
            except Exception as e:
                print(f"❌ 融合服务推荐出错: {e}")
                # 出错时回退到知识图谱
        
        # 如果融合服务不可用或出错，使用真实知识图谱进行推荐
        if self.knowledge_graph is not None:
            print("📊 使用真实知识图谱进行推荐")
            try:
                # 获取知识图谱推荐结果
                kg_recommendations = self.knowledge_graph.get_user_recommendations(user_id, top_k, user_needs)
                
                if kg_recommendations:
                    # 处理知识图谱推荐结果，确保格式统一
                    formatted_recommendations = []
                    for rec in kg_recommendations:
                        formatted_recommendations.append({
                            'product_id': rec['product_id'],
                            'brand': rec['brand'],
                            'category': rec['category'],
                            'price': rec['price'],
                            'sales_count': 0,  # 知识图谱中没有销量信息，设为0
                            'score': rec['score']
                        })
                    return formatted_recommendations
            except Exception as e:
                print(f"❌ 知识图谱推荐出错: {e}")
                # 出错时回退到LightGBM模型
        
        # 如果知识图谱不可用或出错，检查LightGBM模型是否可用
        print("💡 使用LightGBM模型进行推荐")
        if not self.is_trained or self.model is None:
            print("⚠️  LightGBM模型未训练，返回空推荐")
            return []
        
        # 检查user_features是否存在且不为空
        if self.user_features is None or self.user_features.empty:
            print("⚠️  用户特征为空，返回空推荐")
            return []
        
        # 检查商品特征是否存在且不为空
        if self.item_features is None or self.item_features.empty:
            print("⚠️  商品特征为空，返回空推荐")
            return []
        
        # 获取用户特征
        user_feat = self.user_features[self.user_features['user_id'] == user_id]
        if user_feat.empty:
            # 如果用户是新用户，使用默认特征
            user_feat = self.user_features.mean(numeric_only=True).to_frame().T
            user_feat['user_id'] = user_id
            user_feat['favorite_brand'] = 'unknown'  # 默认品牌
            user_feat['favorite_category'] = 'unknown'  # 默认类别
            
            # 添加缺失的行为计数特征
            for behavior in ['pv_count', 'buy_count', 'cart_count', 'collect_count']:
                if behavior not in user_feat.columns:
                    user_feat[behavior] = 0
        
        # 为每个商品生成特征向量
        recommendations = []
        
        for _, item in self.item_features.iterrows():
            # 检查商品是否与用户需求匹配
            if user_needs:
                # 提取用户需求中的关键词
                user_keywords = user_needs.lower().split()
                
                # 提取商品的关键词（从品牌、类别、名称中）
                product_keywords = []
                if 'brand' in item:
                    product_keywords.append(item['brand'].lower())
                if 'category' in item:
                    product_keywords.append(item['category'].lower())
                
                # 定义常见的美妆护肤商品类型映射
                product_type_mapping = {
                    '面霜': ['面霜', '保湿霜', '补水霜', '滋润霜'],
                    '精华': ['精华', '精华液', '原液', '肌底液'],
                    '爽肤水': ['爽肤水', '化妆水', '柔肤水', '收敛水'],
                    '面膜': ['面膜', '补水面膜', '美白面膜', '清洁面膜'],
                    '眼霜': ['眼霜', '眼部精华', '眼膜'],
                    '洁面': ['洁面', '洗面奶', '洁面乳', '洁面膏']
                }
                
                # 检查商品类别是否与用户需求匹配
                is_match = False
                
                # 方法1：直接关键词匹配
                for keyword in user_keywords:
                    if any(keyword in product_keyword for product_keyword in product_keywords):
                        is_match = True
                        break
                
                # 方法2：检查商品类型映射
                if not is_match:
                    for product_type, type_keywords in product_type_mapping.items():
                        # 检查用户需求是否包含该类型的关键词
                        if any(kw in user_needs.lower() for kw in type_keywords):
                            # 检查商品类别是否属于该类型
                            if item.get('category', '').lower() == product_type:
                                is_match = True
                                break
                
                # 如果不匹配，跳过该商品
                if not is_match:
                    continue
            
            # 构建商品特征，确保是DataFrame类型
            item_feat = pd.DataFrame([item])
            
            # 合并用户和商品特征
            user_feat_no_id = user_feat.drop('user_id', axis=1, errors='ignore')
            item_feat_no_id = item_feat.drop('product_id', axis=1, errors='ignore')
            
            combined_feat = pd.concat([user_feat_no_id, item_feat_no_id], axis=1)
            
            # 处理交互计数特征（设置为0，因为是预测）
            interaction_features = ['pv_count', 'buy_count', 'cart_count', 'collect_count', 'interaction_count']
            for feat in interaction_features:
                if feat not in combined_feat.columns:
                    combined_feat[feat] = 0
            
            # 处理分类特征
            for cat_feat in self.feature_meta['categorical_features']:
                if cat_feat in combined_feat.columns:
                    combined_feat[cat_feat] = combined_feat[cat_feat].astype('category')
            
            # 获取训练时使用的所有特征
            train_features = self.feature_meta['numerical_features'] + self.feature_meta['categorical_features']
            
            # 确保combined_feat包含所有训练时的特征
            for feat in train_features:
                if feat not in combined_feat.columns:
                    if feat in self.feature_meta['numerical_features']:
                        combined_feat[feat] = 0
                    elif feat in self.feature_meta['categorical_features']:
                        combined_feat[feat] = 'unknown'
                        combined_feat[feat] = combined_feat[feat].astype('category')
            
            # 只保留训练时使用的特征，并按顺序排列
            combined_feat = combined_feat[train_features]
            
            try:
                # 预测推荐分数
                score = self.model.predict_proba(combined_feat)[0][1]
                
                # 保存推荐结果
                recommendations.append({
                    'product_id': item['product_id'],
                    'brand': item['brand'],
                    'category': item['category'],
                    'price': item['avg_price'],
                    'sales_count': item['sales_count'],
                    'score': float(score)
                })
            except Exception as e:
                print(f"预测商品 {item['product_id']} 时出错: {e}")
                continue
        
        # 按推荐分数排序，返回top_k结果
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # 如果推荐结果为空，使用本地商品数据作为备用
        if not recommendations and self.local_product_data:
            print("💡 使用本地商品数据作为备用推荐")
            # 基于用户需求过滤本地商品数据
            filtered_products = []
            if user_needs:
                user_keywords = user_needs.lower().split()
                for product in self.local_product_data:
                    # 检查商品是否与用户需求匹配
                    product_text = f"{product['name']} {product['brand']} {product['category']}".lower()
                    if any(keyword in product_text for keyword in user_keywords):
                        filtered_products.append(product)
            else:
                filtered_products = self.local_product_data
            
            # 如果过滤后有结果，使用过滤后的结果；否则使用所有本地商品数据
            if filtered_products:
                backup_recommendations = []
                for product in filtered_products[:top_k]:
                    backup_recommendations.append({
                        'product_id': product['product_id'],
                        'brand': product['brand'],
                        'category': product['category'],
                        'price': product['price'],
                        'sales_count': 0,
                        'score': 0.5  # 默认分数
                    })
                print(f"✅ 使用本地商品数据生成了 {len(backup_recommendations)} 条推荐")
                return backup_recommendations
            else:
                print("❌ 本地商品数据也没有找到匹配的结果")
        
        return recommendations
    
    def _load_local_product_data(self) -> None:
        """
        加载本地商品数据作为备用
        """
        try:
            import os
            
            # 加载CSV数据
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "face_data", "face_goods_info.csv")
            if not os.path.exists(data_path):
                print(f"❌ 数据文件不存在: {data_path}")
                return
            
            print(f"从 {data_path} 加载本地美妆商品数据作为备用...")
            data = pd.read_csv(data_path)
            
            print(f"CSV文件列名: {list(data.columns)}")
            print(f"CSV文件行数: {len(data)}")
            
            # 提取数据
            for _, row in data.iterrows():
                try:
                    product_id = str(row['ID'])
                    if not product_id:
                        continue
                    
                    # 构建文档内容
                    name = row['名称']
                    brand = row['商家']
                    category = row['品类']
                    price = row['单价']
                    
                    self.local_product_data.append({
                        'product_id': product_id,
                        'name': name,
                        'brand': brand,
                        'category': category,
                        'price': price
                    })
                except Exception as e:
                    print(f"处理行数据失败: {e}")
                    continue
            
            print(f"✅ 成功加载 {len(self.local_product_data)} 个本地美妆商品数据作为备用")
        except Exception as e:
            print(f"❌ 加载本地商品数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _populate_vector_db_directly(self) -> None:
        """
        直接填充向量数据库
        """
        try:
            import os
            import chromadb
            from chromadb.config import Settings
            
            # 设置向量数据库路径
            vdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_vector_db")
            collection_name = "face_recommendation"
            
            print(f"直接填充向量数据库路径: {vdb_path}")
            
            # 初始化向量数据库客户端
            client = chromadb.PersistentClient(
                path=vdb_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取或创建集合
            collection = client.get_or_create_collection(name=collection_name)
            
            # 检查文档数量
            doc_count = collection.count()
            print(f"当前向量数据库文档数量: {doc_count}")
            
            if doc_count > 0:
                print("✅ 向量数据库已有数据，跳过填充")
                return
            
            # 加载CSV数据
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "face_data", "face_goods_info.csv")
            if not os.path.exists(data_path):
                print(f"❌ 数据文件不存在: {data_path}")
                return
            
            print(f"从 {data_path} 加载美妆商品数据...")
            data = pd.read_csv(data_path)
            
            print(f"CSV文件列名: {list(data.columns)}")
            print(f"CSV文件行数: {len(data)}")
            
            # 提取数据
            ids = []
            documents = []
            metadatas = []
            
            for _, row in data.iterrows():
                try:
                    product_id = str(row['ID'])
                    if not product_id:
                        continue
                    
                    # 构建文档内容
                    name = row['名称']
                    brand = row['商家']
                    category = row['品类']
                    price = row['单价']
                    
                    document = f"{name} {brand} {category}"
                    
                    ids.append(product_id)
                    documents.append(document)
                    metadatas.append({
                        'product_id': product_id,
                        'name': name,
                        'brand': brand,
                        'category': category,
                        'price': price
                    })
                except Exception as e:
                    print(f"处理行数据失败: {e}")
                    continue
            
            if ids:
                print(f"准备添加 {len(ids)} 个商品到向量数据库")
                # 分批添加数据
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_documents = documents[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    print(f"添加批次 {i//batch_size + 1}/{(len(ids)+batch_size-1)//batch_size}，共 {len(batch_ids)} 个商品")
                    collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    print(f"批次 {i//batch_size + 1} 添加成功")
                
                # 验证添加结果
                doc_count = collection.count()
                print(f"✅ 成功填充 {doc_count} 个美妆商品到向量数据库")
            else:
                print("❌ 没有从CSV文件中提取到有效的商品数据")
                
        except Exception as e:
            print(f"❌ 直接填充向量数据库失败: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        离线评估美妆护肤推荐模型
        
        Args:
            test_data: 测试数据
        
        Returns:
            Dict[str, float]: 评估指标
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 分离特征和标签
        X_test = test_data.drop(['label', 'user_id', 'product_id'], axis=1, errors='ignore')
        y_test = test_data['label']
        
        # 预测概率
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算评估指标
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # 计算NDCG
        # 这里简化处理，实际NDCG计算需要考虑用户的真实排序
        ndcg = ndcg_score([y_test.values], [y_pred_proba])
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'ndcg': float(ndcg),
            'auc': float(auc)
        }

if __name__ == "__main__":
    # 示例用法
    module = FaceRecommendationModule()
    
    # 加载商品数据
    data_path = os.path.join("data", "face_data", "face_goods_info.csv")
    data = module.load_data(data_path)
    preprocessed_data = module.preprocess_data(data)
    
    # 特征工程
    train_data, feature_meta = module.feature_engineering(preprocessed_data)
    
    # 训练模型
    module.train(train_data)
    
    # 生成推荐
    recommendations = module.predict({
        'user_id': 'user_1',
        'user_needs': '我需要一款补水保湿的面霜'
    }, top_k=5)
    
    print("\n=== 美妆护肤推荐结果 ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['brand']} {rec['category']} - 价格：{rec['price']}元 - 推荐分数：{rec['score']:.4f}")
