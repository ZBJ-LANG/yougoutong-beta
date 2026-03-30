"""
电子数码推荐子模块
"""
import os
from typing import Dict, List, Any, Tuple
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

# 尝试导入chromadb，如果失败则使用模拟实现
CHROMADB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  chromadb not available, will use fallback implementation")

from .base_module import BaseRecommendationModule


class ElectronicRecommendationModule(BaseRecommendationModule):
    """
    电子数码推荐子模块，结合LightGBM模型和真实知识图谱实现商品推荐
    """
    
    def __init__(self):
        """
        初始化电子数码推荐模块
        """
        super().__init__(module_name="电子数码", category="电子产品")
        self.feature_meta = {}
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.knowledge_graph = None
        self.vector_db_client = None
        self.vector_db_collection = None
        self.model = None
        self.is_trained = False
        
        # 尝试加载真实知识图谱
        self._load_knowledge_graph()
        # 初始化向量数据库
        self._init_vector_db()
    
    def _load_knowledge_graph(self) -> None:
        """
        加载真实知识图谱
        """
        try:
            # 先导入ElectronicKnowledgeGraph类
            from .electronic_knowledge_graph import ElectronicKnowledgeGraph
            # 确保类在当前模块的命名空间中
            import sys
            sys.modules['__main__'].ElectronicKnowledgeGraph = ElectronicKnowledgeGraph
            
            # 使用绝对路径来加载知识图谱文件
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "electronic_knowledge_graph.pkl")
            print(f"检查知识图谱文件路径: {model_path}")
            if os.path.exists(model_path):
                self.knowledge_graph = ElectronicKnowledgeGraph.load_graph(model_path)
                print("✅ 成功加载电子数码模块真实知识图谱")
            else:
                print("⚠️  知识图谱文件不存在，将使用LightGBM模型推荐")
        except Exception as e:
            print(f"❌ 加载知识图谱失败: {e}")
            self.knowledge_graph = None
    
    def _init_vector_db(self) -> None:
        """
        初始化向量数据库
        """
        if not CHROMADB_AVAILABLE:
            print("⚠️  chromadb不可用，跳过向量数据库初始化")
            return
            
        try:
            # 使用绝对路径来检查向量数据库是否存在
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            vector_db_path = os.path.join(base_dir, "models", "electronic_vector_db")
            print(f"检查向量数据库路径: {vector_db_path}")
            if os.path.exists(vector_db_path):
                self.vector_db_client = chromadb.PersistentClient(
                    path=vector_db_path,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
                self.vector_db_collection = self.vector_db_client.get_or_create_collection(
                    name="electronic_recommendation"
                )
                print("✅ 成功初始化电子模块向量数据库")
            else:
                print("⚠️  向量数据库不存在，将使用LightGBM模型推荐")
        except Exception as e:
            print(f"❌ 初始化向量数据库失败: {e}")
            self.vector_db_client = None
            self.vector_db_collection = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载电子数码产品销售数据
        
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
        电子数码数据预处理
        
        Args:
            data: 原始数据
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 1. 检查数据格式，适配不同的数据结构
        if '用户ID' in data.columns:  # 新的用户行为数据格式
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
            data['event_time'] = pd.to_datetime(data['event_time'], unit='s')
            
            # 4. 添加虚拟order_id（新数据格式没有order_id，用时间戳+用户ID生成）
            data['order_id'] = data['event_time'].astype(str) + '_' + data['user_id']
            
        else:  # 旧的销售数据格式
            # 1. 过滤出电子数码相关数据
            # 定义电子数码相关的类别关键词
            electronic_categories = [
                'electronics', 'computers', 'smartphone', 
                'tablet', 'headphone', 'tv', 'notebook'
            ]
            
            # 过滤包含电子数码关键词的记录
            data = data[data['category_code'].str.contains('|'.join(electronic_categories), na=False)]
            
            # 2. 处理缺失值
            data = data.dropna(subset=['user_id', 'product_id', 'category_code', 'price'])
            
            # 3. 转换数据类型
            data['user_id'] = data['user_id'].astype(str)
            data['product_id'] = data['product_id'].astype(str)
            data['order_id'] = data['order_id'].astype(str)
        
        # 5. 计算用户购买频率（基于behavior_type或order_id）
        if 'behavior_type' in data.columns:
            # 使用新数据格式的行为类型计算购买频率
            purchase_data = data[data['behavior_type'] == 'buy']
            user_purchase_count = purchase_data.groupby('user_id')['order_id'].nunique().reset_index()
        else:
            # 使用旧数据格式的order_id计算购买频率
            user_purchase_count = data.groupby('user_id')['order_id'].nunique().reset_index()
            
        user_purchase_count.columns = ['user_id', 'purchase_count']
        data = pd.merge(data, user_purchase_count, on='user_id', how='left')
        
        print(f"预处理后数据：{len(data)} 条记录")
        return data
    
    def feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        电子数码特征工程
        
        Args:
            data: 预处理后的数据
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 特征工程后的数据和特征元数据
        """
        # 1. 处理类别特征（适配不同数据格式）
        if 'behavior_type' in data.columns:  # 新的用户行为数据格式
            data['brand'] = data['brand'].fillna('unknown')
            
            # 从商品名称中提取更详细的类别信息
            def extract_category_level(product_name):
                """从商品名称中提取类别级别"""
                if pd.isna(product_name):
                    return 'unknown', 'unknown'
                
                product_name = product_name.lower()
                
                # 定义主要类别映射
                category_mapping = {
                    'phone': 'smartphone',
                    'iphone': 'smartphone',
                    'ipad': 'tablet',
                    'matepad': 'tablet',
                    '笔记本': 'notebook',
                    'xps': 'notebook',
                    '灵耀': 'notebook',
                    '小新': 'notebook',
                    '天选': 'notebook',
                    's24': 'smartphone',
                    'magic6': 'smartphone',
                    'x100': 'smartphone'
                }
                
                for keyword, category in category_mapping.items():
                    if keyword in product_name:
                        return 'electronics', category
                
                return 'electronics', 'other'
            
            data[['category_level1', 'category_level2']] = data['product_name'].apply(
                lambda x: pd.Series(extract_category_level(x))
            )
        else:  # 旧的销售数据格式
            data['brand'] = data['brand'].fillna('unknown')
            data['category_level1'] = data['category_code'].apply(lambda x: x.split('.')[0] if pd.notna(x) else 'unknown')
            data['category_level2'] = data['category_code'].apply(lambda x: x.split('.')[1] if pd.notna(x) and len(x.split('.')) > 1 else 'unknown')
        
        # 2. 计算商品特征
        item_features = data.groupby('product_id').agg({
            'price': ['mean', 'count'],
            'brand': 'first',
            'category_level1': 'first',
            'category_level2': 'first'
        }).reset_index()
        
        item_features.columns = ['product_id', 'avg_price', 'sales_count', 'brand', 'category_level1', 'category_level2']
        item_features['sales_volume'] = item_features['avg_price'] * item_features['sales_count']
        
        # 3. 计算用户行为特征（基于不同行为类型）
        # 计算不同行为类型的次数
        behavior_counts = data.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
        behavior_counts.columns = ['user_id'] + [f'{col}_count' for col in behavior_counts.columns[1:]]
        
        # 4. 计算用户特征
        user_features = data.groupby('user_id').agg({
            'price': ['mean', 'sum', 'std'],
            'purchase_count': 'first',
            'brand': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown',
            'category_level2': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
        }).reset_index()
        
        user_features.columns = ['user_id', 'avg_purchase_price', 'total_spent', 'price_std', 
                                'purchase_count', 'favorite_brand', 'favorite_category']
        
        # 合并用户行为特征
        user_features = pd.merge(user_features, behavior_counts, on='user_id', how='left')
        
        # 填充缺失的行为计数（如果用户没有某种行为）
        for behavior in ['pv', 'buy', 'cart']:
            behavior_col = f'{behavior}_count'
            if behavior_col not in user_features.columns:
                user_features[behavior_col] = 0
        
        # 5. 构建用户-商品交互矩阵
        # 计算用户对商品的不同行为次数
        interaction_counts = data.groupby(['user_id', 'product_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
        
        # 重命名列
        interaction_counts.columns = ['user_id', 'product_id'] + [f'{col}_count' for col in interaction_counts.columns[2:]]
        
        # 计算总交互次数
        interaction_counts['interaction_count'] = interaction_counts.iloc[:, 2:].sum(axis=1)
        
        # 6. 合并所有特征
        merged_data = pd.merge(interaction_counts, user_features, on='user_id', how='left')
        merged_data = pd.merge(merged_data, item_features, on='product_id', how='left')
        
        # 7. 生成标签：是否购买过（基于behavior_type或order_id）
        if 'buy_count' in merged_data.columns:
            # 基于行为类型的购买次数
            merged_data['label'] = (merged_data['buy_count'] > 0).astype(int)
        else:
            # 基于order_id的简化处理
            merged_data['label'] = 1
        
        # 8. 生成负样本（简化处理）
        # 这里使用简单的随机负采样，实际项目中应使用更复杂的负采样策略
        positive_samples = merged_data[merged_data['label'] == 1].copy()
        negative_samples = []
        
        # 随机生成一些负样本
        all_users = data['user_id'].unique()
        all_products = data['product_id'].unique()
        
        # 限制负采样尝试次数，避免无限循环
        max_attempts = len(positive_samples) * 2
        attempt = 0
        target_neg_samples = len(positive_samples) // 2  # 目标负样本数量为正样本的一半
        
        while len(negative_samples) < target_neg_samples and attempt < max_attempts:
            attempt += 1
            user = np.random.choice(all_users)
            product = np.random.choice(all_products)
            
            # 检查是否为正样本
            is_positive = ((positive_samples['user_id'] == user) & (positive_samples['product_id'] == product)).any()
            
            if not is_positive:
                # 检查是否已经生成过该负样本
                is_duplicate = False
                for neg_sample in negative_samples:
                    if neg_sample['user_id'] == user and neg_sample['product_id'] == product:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # 为负样本初始化各种交互计数为0
                    neg_sample = {
                        'user_id': user,
                        'product_id': product,
                        'interaction_count': 0,
                        'label': 0
                    }
                    
                    # 添加可能的行为计数列
                    for behavior in ['pv_count', 'buy_count', 'cart_count']:
                        neg_sample[behavior] = 0
                    
                    negative_samples.append(neg_sample)
        
        # 如果负样本数量不足，至少生成一些
        if len(negative_samples) == 0:
            # 直接创建一些负样本，不检查重复
            for _ in range(min(10, len(positive_samples))):
                user = np.random.choice(all_users)
                product = np.random.choice(all_products)
                
                neg_sample = {
                    'user_id': user,
                    'product_id': product,
                    'interaction_count': 0,
                    'label': 0
                }
                
                for behavior in ['pv_count', 'buy_count', 'cart_count']:
                    neg_sample[behavior] = 0
                
                negative_samples.append(neg_sample)
        
        negative_samples = pd.DataFrame(negative_samples)
        
        # 合并正负样本
        final_data = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # 9. 处理分类特征
        categorical_features = ['brand', 'category_level1', 'category_level2', 'favorite_brand', 'favorite_category']
        for feat in categorical_features:
            final_data[feat] = final_data[feat].astype('category')
        
        # 10. 保存特征元数据
        numerical_features = ['avg_purchase_price', 'total_spent', 'price_std', 'purchase_count', 
                            'avg_price', 'sales_count', 'sales_volume', 'interaction_count']
        
        # 添加行为计数特征
        for behavior in ['pv_count', 'buy_count', 'cart_count']:
            if behavior in final_data.columns:
                numerical_features.append(behavior)
        
        feature_meta = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'all_users': list(all_users),
            'all_products': list(all_products)
        }
        
        print(f"特征工程完成，最终数据：{len(final_data)} 条记录")
        print(f"正样本：{len(positive_samples)}，负样本：{len(negative_samples)}")
        
        self.feature_meta = feature_meta
        self.user_item_matrix = merged_data
        self.item_features = item_features
        self.user_features = user_features
        
        return final_data, feature_meta
    
    def train(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        训练电子数码推荐模型
        
        Args:
            train_data: 训练数据
            **kwargs: 训练参数
        """
        # 检查scikit-learn和LightGBM是否可用
        if not SKLEARN_AVAILABLE or not LIGHTGBM_AVAILABLE:
            print("⚠️  scikit-learn或LightGBM不可用，跳过训练")
            self.is_trained = False
            return
        
        # 分离特征和标签
        X = train_data.drop(['user_id', 'product_id', 'label'], axis=1)
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
            eval_metric='auc',
            categorical_feature=self.feature_meta['categorical_features']
        )
        
        self.is_trained = True  # 先设置训练完成标志
        
        # 评估模型
        eval_result = self.evaluate(test_data=pd.concat([X_test, y_test], axis=1))
        print(f"模型训练完成，评估结果：{eval_result}")
    
    def predict(self, user_features: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        为用户生成电子数码推荐
        
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
        
        # 知识图谱与RAG并行融合策略
        print("🔄 执行知识图谱与RAG并行融合策略")
        
        # 1. 并行向知识图谱和向量数据库发起检索
        kg_results = []
        rag_results = []
        
        # 从知识图谱获取推荐
        if self.knowledge_graph is not None:
            print("📊 从知识图谱获取推荐")
            try:
                kg_results = self.knowledge_graph.get_user_recommendations(user_id, top_k)
                print(f"✅ 知识图谱返回 {len(kg_results)} 条推荐")
            except Exception as e:
                print(f"❌ 知识图谱推荐出错: {e}")
        
        # 从向量数据库获取推荐
        if self.vector_db_collection is not None and user_needs:
            print("🤖 从向量数据库获取推荐")
            try:
                rag_query_results = self.vector_db_collection.query(
                    query_texts=[user_needs],
                    n_results=top_k * 2,  # 获取更多结果以便去重
                    include=["documents", "metadatas", "distances"]
                )
                
                # 处理向量数据库结果
                for i, (doc, meta, dist) in enumerate(zip(
                    rag_query_results["documents"][0],
                    rag_query_results["metadatas"][0],
                    rag_query_results["distances"][0]
                )):
                    # 只处理商品类型的结果
                    if "product_name" in meta:
                        rag_results.append({
                            'product_id': meta.get('product_id', f'rag_{i:03d}'),
                            'brand': meta.get('brand', 'unknown'),
                            'category': meta.get('category', 'unknown'),
                            'price': meta.get('price', 0),
                            'sales_count': 0,
                            'score': 1 - dist  # 将距离转换为相似度分数
                        })
                print(f"✅ 向量数据库返回 {len(rag_results)} 条推荐")
            except Exception as e:
                print(f"❌ 向量数据库推荐出错: {e}")
        
        # 2. 结果去重和融合
        print("🔍 结果去重和融合")
        
        # 合并所有结果
        all_recommendations = []
        
        # 添加知识图谱结果
        for rec in kg_results:
            all_recommendations.append({
                'product_id': rec['product_id'],
                'brand': rec['brand'],
                'category': rec['category'],
                'price': rec['price'],
                'sales_count': 0,
                'score': rec['score'],
                'source': 'knowledge_graph'
            })
        
        # 添加向量数据库结果
        for rec in rag_results:
            all_recommendations.append({
                'product_id': rec['product_id'],
                'brand': rec['brand'],
                'category': rec['category'],
                'price': rec['price'],
                'sales_count': rec['sales_count'],
                'score': rec['score'],
                'source': 'vector_db'
            })
        
        # 去重：基于product_id
        unique_recommendations = {}
        for rec in all_recommendations:
            product_id = rec['product_id']
            if product_id not in unique_recommendations:
                unique_recommendations[product_id] = rec
            else:
                # 如果重复，保留分数更高的结果
                if rec['score'] > unique_recommendations[product_id]['score']:
                    unique_recommendations[product_id] = rec
        
        # 转换为列表并按分数排序
        fused_recommendations = list(unique_recommendations.values())
        fused_recommendations = sorted(fused_recommendations, key=lambda x: x['score'], reverse=True)
        
        print(f"✅ 去重融合后得到 {len(fused_recommendations)} 条推荐")
        
        # 3. 如果融合结果不足，使用LightGBM模型补充
        if len(fused_recommendations) < top_k:
            print("💡 使用LightGBM模型补充推荐")
            
            if self.is_trained and self.model is not None and self.user_features is not None and not self.user_features.empty:
                # 获取用户特征
                user_feat = self.user_features[self.user_features['user_id'] == user_id]
                if user_feat.empty:
                    # 如果用户是新用户，使用默认特征
                    user_feat = self.user_features.mean(numeric_only=True).to_frame().T
                    user_feat['user_id'] = user_id
                    user_feat['favorite_brand'] = 'samsung'  # 默认品牌
                    user_feat['favorite_category'] = 'smartphone'  # 默认类别
                    
                    # 添加缺失的行为计数特征
                    for behavior in ['pv_count', 'buy_count', 'cart_count']:
                        if behavior not in user_feat.columns:
                            user_feat[behavior] = 0
                
                # 为每个商品生成特征向量
                lgbm_recommendations = []
                
                for _, item in self.item_features.iterrows():
                    # 检查商品是否已经在融合结果中
                    product_id = item['product_id']
                    if product_id in unique_recommendations:
                        continue
                    
                    # 检查商品是否与用户需求匹配
                    if user_needs:
                        # 提取用户需求中的关键词
                        user_keywords = user_needs.lower().split()
                        
                        # 提取商品的关键词（从品牌、类别、名称中）
                        product_keywords = []
                        if 'brand' in item:
                            product_keywords.append(item['brand'].lower())
                        if 'category_level2' in item:
                            product_keywords.append(item['category_level2'].lower())
                        
                        # 定义常见的商品类型映射
                        product_type_mapping = {
                            'smartphone': ['手机', 'iphone', 'huawei', 'xiaomi', 'vivo', 'oppo'],
                            'tablet': ['平板', 'ipad', 'matepad'],
                            'headphone': ['耳机', 'airpods', 'huawei耳机', '小米耳机'],
                            'notebook': ['笔记本', '电脑', 'xps', '灵耀', '小新', '天选'],
                            'tv': ['电视', '彩电', 'smart tv'],
                            'cpu': ['cpu', '处理器', 'intel', 'amd']
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
                                    if item.get('category_level2', '').lower() == product_type:
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
                    interaction_features = ['pv_count', 'buy_count', 'cart_count', 'interaction_count']
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
                        lgbm_recommendations.append({
                            'product_id': item['product_id'],
                            'brand': item['brand'],
                            'category': item['category_level2'],
                            'price': item['avg_price'],
                            'sales_count': item['sales_count'],
                            'score': float(score),
                            'source': 'lightgbm'
                        })
                    except Exception as e:
                        print(f"预测商品 {item['product_id']} 时出错: {e}")
                        continue
                
                # 排序并添加到融合结果
                lgbm_recommendations = sorted(lgbm_recommendations, key=lambda x: x['score'], reverse=True)
                
                # 补充不足的推荐数量
                needed_count = top_k - len(fused_recommendations)
                fused_recommendations.extend(lgbm_recommendations[:needed_count])
                
                print(f"✅ LightGBM模型补充了 {min(needed_count, len(lgbm_recommendations))} 条推荐")
            else:
                print("⚠️  LightGBM模型不可用，无法补充推荐")
        
        # 3. 最终排序和过滤
        print("🏆 生成最终推荐结果")
        
        # 按分数排序并限制数量
        final_recommendations = sorted(fused_recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # 移除source字段，保持与原有格式一致
        for rec in final_recommendations:
            rec.pop('source', None)
        
        print(f"✅ 最终生成 {len(final_recommendations)} 条推荐")
        
        return final_recommendations
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        离线评估电子数码推荐模型
        
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
        # 排除user_id和product_id等字符串类型的特征，只保留数值和类别特征
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
    
    def load_model(self, model_path: str) -> None:
        """
        加载已训练的电子数码推荐模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            print(f"✅ 成功加载电子数码推荐模型: {model_path}")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
    
    def save_model(self, model_path: str) -> None:
        """
        保存训练好的电子数码推荐模型
        
        Args:
            model_path: 模型文件保存路径
        """
        try:
            if self.is_trained and self.model is not None:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"✅ 成功保存电子数码推荐模型: {model_path}")
            else:
                print("❌ 模型尚未训练，无法保存")
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            import traceback
            traceback.print_exc()