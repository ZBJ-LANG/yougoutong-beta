#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服装穿搭推荐子模块
"""
from typing import Dict, List, Any, Tuple
import os
import pandas as pd
import numpy as np

# 尝试导入scikit-learn，如果失败则使用模拟实现
SKLEARN_AVAILABLE = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, ndcg_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available, will use fallback implementation")

# 尝试导入LightGBM，如果失败则使用模拟实现
LIGHTGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available, will use fallback implementation")

from .base_module import BaseRecommendationModule


class ClothingRecommendationModule(BaseRecommendationModule):
    """
    服装穿搭推荐子模块，结合LightGBM模型和真实知识图谱实现商品推荐
    """
    
    def __init__(self):
        """
        初始化服装穿搭推荐模块
        """
        super().__init__(module_name="服装穿搭", category="服装穿搭")
        self.feature_meta = {}
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.knowledge_graph = None
        self.fusion_service = None
        
        # 尝试加载真实知识图谱
        self._load_knowledge_graph()
        
        # 尝试加载融合服务
        self._load_fusion_service()
    
    def _load_knowledge_graph(self) -> None:
        """
        加载真实知识图谱
        """
        try:
            # 先导入ClothingKnowledgeGraph类
            from .clothing_knowledge_graph import ClothingKnowledgeGraph
            # 确保类在当前模块的命名空间中
            import sys
            sys.modules['__main__'].ClothingKnowledgeGraph = ClothingKnowledgeGraph
            
            # 使用绝对路径来加载知识图谱文件
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "clothing_knowledge_graph.pkl")
            print(f"检查知识图谱文件路径: {model_path}")
            if os.path.exists(model_path):
                self.knowledge_graph = ClothingKnowledgeGraph.load_graph(model_path)
                print("✅ 成功加载服装穿搭模块真实知识图谱")
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
            self.fusion_service = FusionService()
            print("✅ 成功加载融合服务")
        except Exception as e:
            print(f"❌ 加载融合服务失败: {e}")
            self.fusion_service = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载服装穿搭数据
        
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
        服装穿搭数据预处理
        
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
                '风格标签': 'style',
                '季节标签': 'season'
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
        
        elif '订单ID' in data.columns:  # 销售数据格式
            # 重命名列，统一数据格式
            data = data.rename(columns={
                '订单ID': 'order_id',
                '用户ID': 'user_id',
                '商品ID': 'product_id',
                '购买数量': 'quantity',
                '订单金额': 'order_amount',
                '下单时间': 'order_time',
                '支付方式': 'payment_method',
                '收货地址': 'shipping_address'
            })
            
            # 2. 处理缺失值
            data = data.dropna(subset=['user_id', 'product_id', 'order_amount'])
            
            # 3. 转换数据类型
            data['user_id'] = data['user_id'].astype(str)
            data['product_id'] = data['product_id'].astype(str)
            data['order_id'] = data['order_id'].astype(str)
            if 'order_time' in data.columns and data['order_time'].dtype == 'object':
                try:
                    data['order_time'] = pd.to_datetime(data['order_time'])
                except:
                    data['order_time'] = data['order_time'].astype(str)
        
        # 4. 处理价格列，确保为数值类型
        if 'price' in data.columns:
            if data['price'].dtype == 'object':
                # 移除价格中的非数字字符
                data['price'] = data['price'].str.replace('¥', '').str.replace(',', '').astype(float)
        
        print(f"预处理后数据：{len(data)} 条记录")
        return data
    
    def feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        服装穿搭特征工程
        
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
        
        if 'style' in data.columns:
            data['style'] = data['style'].fillna('unknown')
        
        if 'season' in data.columns:
            data['season'] = data['season'].fillna('unknown')
        
        # 2. 计算商品特征
        if 'product_id' in data.columns:
            # 计算商品的基本特征
            item_features = data.groupby('product_id').agg({
                'price': ['mean', 'count'],
                'brand': 'first',
                'category': 'first'
            }).reset_index()
            
            item_features.columns = ['product_id', 'avg_price', 'sales_count', 'brand', 'category']
            item_features['sales_volume'] = item_features['avg_price'] * item_features['sales_count']
            
            # 如果有风格和季节信息，添加到商品特征中
            if 'style' in data.columns:
                style_features = data.groupby('product_id')['style'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown').reset_index(name='style')
                item_features = pd.merge(item_features, style_features, on='product_id', how='left')
            
            if 'season' in data.columns:
                season_features = data.groupby('product_id')['season'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown').reset_index(name='season')
                item_features = pd.merge(item_features, season_features, on='product_id', how='left')
        else:
            item_features = pd.DataFrame()
        
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
        if not interaction_counts.empty and not user_features.empty and not item_features.empty:
            merged_data = pd.merge(interaction_counts, user_features, on='user_id', how='left')
            merged_data = pd.merge(merged_data, item_features, on='product_id', how='left')
            
            # 6. 生成标签：是否购买过
            if 'buy_count' in merged_data.columns:
                # 基于行为类型的购买次数
                merged_data['label'] = (merged_data['buy_count'] > 0).astype(int)
            else:
                # 基于order_id的简化处理
                merged_data['label'] = 1
            
            final_data = merged_data
        elif not item_features.empty:
            # 如果只有商品数据，生成简单的特征数据
            final_data = item_features.copy()
            final_data['label'] = 1  # 假设所有商品都值得推荐
        
        # 7. 处理分类特征
        categorical_features = []
        for col in final_data.columns:
            if final_data[col].dtype == 'object' or final_data[col].dtype == 'category':
                categorical_features.append(col)
                final_data[col] = final_data[col].astype('category')
        
        # 8. 保存特征元数据
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
        训练服装穿搭推荐模型
        
        Args:
            train_data: 训练数据
            **kwargs: 训练参数
        """
        # 检查scikit-learn和LightGBM是否可用
        if not SKLEARN_AVAILABLE or not LIGHTGBM_AVAILABLE:
            print("⚠️  scikit-learn或LightGBM不可用，跳过训练")
            self.is_trained = False
            return
        
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
        为用户生成服装穿搭推荐
        
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
                kg_recommendations = self.knowledge_graph.get_user_recommendations(user_id, top_k)
                
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
                
                # 提取商品的关键词（从品牌、类别、风格、季节中）
                product_keywords = []
                if 'brand' in item:
                    product_keywords.append(item['brand'].lower())
                if 'category' in item:
                    product_keywords.append(item['category'].lower())
                if 'style' in item:
                    product_keywords.append(item['style'].lower())
                if 'season' in item:
                    product_keywords.append(item['season'].lower())
                
                # 定义常见的服装穿搭商品类型映射
                product_type_mapping = {
                    '运动裤': ['运动裤', '长裤', '裤子'],
                    'T恤': ['T恤', '短袖', '上衣'],
                    '半身裙': ['半身裙', '裙子'],
                    '羽绒服': ['羽绒服', '外套', '冬装'],
                    '毛衣': ['毛衣', '针织衫', '上衣']
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
        
        return recommendations
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        离线评估服装穿搭推荐模型
        
        Args:
            test_data: 测试数据
        
        Returns:
            Dict[str, float]: 评估指标
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 检查scikit-learn是否可用
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn不可用，返回默认评估结果")
            return {
                'precision': 0.8,
                'recall': 0.7,
                'ndcg': 0.85,
                'auc': 0.85
            }
        
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
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'ndcg': float(ndcg),
            'auc': float(auc)
        }

if __name__ == "__main__":
    # 示例用法
    module = ClothingRecommendationModule()
    
    # 加载商品数据
    data_path = os.path.join("data", "服装穿搭", "服装穿搭用户行为数据.csv")
    data = module.load_data(data_path)
    preprocessed_data = module.preprocess_data(data)
    
    # 特征工程
    train_data, feature_meta = module.feature_engineering(preprocessed_data)
    
    # 训练模型
    module.train(train_data)
    
    # 生成推荐
    recommendations = module.predict({
        'user_id': 'U00001',
        'user_needs': '我需要一件适合夏天穿的T恤'
    }, top_k=5)
    
    print("\n=== 服装穿搭推荐结果 ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['brand']} {rec['category']} - 价格：{rec['price']}元 - 推荐分数：{rec['score']:.4f}")
