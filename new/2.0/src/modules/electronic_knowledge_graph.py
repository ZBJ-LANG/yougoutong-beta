#!/usr/bin/env python3
"""
电子模块知识图谱构建模块
"""
import os
import pandas as pd
import networkx as nx
import pickle
import json
from typing import Dict, List, Any

class ElectronicKnowledgeGraph:
    """
    电子模块真实知识图谱
    基于电子产品用户行为数据构建
    """
    
    def __init__(self, kg_json_path=None):
        """初始化知识图谱
        
        Args:
            kg_json_path: 知识图谱JSON文件路径
        """
        self.graph = nx.DiGraph()
        self.kg_json_path = kg_json_path
        self.entities = {
            'User': set(),
            'Product': set(),
            'Brand': set(),
            'Category': set(),
            'TechnicalParameter': set(),
            'UsageScenario': set(),
            'UserGroup': set(),
            'Accessory': set()
        }
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载电子产品用户行为数据"""
        print(f"正在加载数据: {data_path}")
        data = pd.read_csv(data_path, encoding='utf-8')
        print(f"成功加载 {len(data)} 条记录")
        return data
    
    def load_from_json(self):
        """从JSON文件加载知识图谱数据"""
        if not self.kg_json_path:
            print("❌ 未指定知识图谱JSON文件路径")
            return False
        
        if not os.path.exists(self.kg_json_path):
            print(f"❌ JSON文件 {self.kg_json_path} 不存在")
            return False
        
        print(f"从 {self.kg_json_path} 加载知识图谱数据...")
        
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            
            # 加载实体
            for entity_type, entities in kg_data.get('entities', {}).items():
                if entity_type == 'Brand':
                    for entity in entities:
                        self.entities['Brand'].add(entity['name'])
                elif entity_type == 'Product':
                    for entity in entities:
                        self.entities['Product'].add(entity['name'])
                elif entity_type == 'Category':
                    for entity in entities:
                        self.entities['Category'].add(entity['name'])
                elif entity_type == 'TechnicalParameter':
                    for entity in entities:
                        self.entities['TechnicalParameter'].add(entity['name'])
                elif entity_type == 'UsageScenario':
                    for entity in entities:
                        self.entities['UsageScenario'].add(entity['name'])
                elif entity_type == 'UserGroup':
                    for entity in entities:
                        self.entities['UserGroup'].add(entity['name'])
                elif entity_type == 'Accessory':
                    for entity in entities:
                        self.entities['Accessory'].add(entity['name'])
            
            # 加载关系
            for relation in kg_data.get('relations', []):
                source = relation['source']
                target = relation['target']
                rel_type = relation['type']
                
                # 添加节点
                if source not in self.graph.nodes:
                    # 推断节点类型
                    node_type = self._infer_entity_type(source)
                    self.graph.add_node(source, entity_type=node_type)
                if target not in self.graph.nodes:
                    node_type = self._infer_entity_type(target)
                    self.graph.add_node(target, entity_type=node_type)
                
                # 添加边
                relation_map = {
                    'has_parameter': {'relation': '具有', 'type': '技术参数关系'},
                    'suitable_for': {'relation': '适合', 'type': '适用场景关系'},
                    'targets': {'relation': '针对', 'type': '用户群体关系'},
                    'compatible_with': {'relation': '兼容', 'type': '配件兼容关系'},
                    'belongs_to': {'relation': '属于', 'type': '品牌关系'},
                    'category': {'relation': '属于', 'type': '类别关系'}
                }
                
                edge_data = relation_map.get(rel_type, {'relation': rel_type, 'type': '其他关系'})
                self.graph.add_edge(source, target, relation=edge_data['relation'], type=edge_data['type'])
            
            print(f"✅ JSON数据加载完成")
            print(f"  - 节点数量：{self.graph.number_of_nodes()}")
            print(f"  - 边数量：{self.graph.number_of_edges()}")
            return True
        except Exception as e:
            print(f"❌ 加载JSON文件失败：{e}")
            return False
    
    def _infer_entity_type(self, entity_name):
        """推断实体类型"""
        for entity_type, entities in self.entities.items():
            if entity_name in entities:
                return entity_type
        return '未知'
    
    def extract_entities(self, data: pd.DataFrame) -> None:
        """从数据中抽取实体"""
        print("正在抽取实体...")
        
        # 抽取用户实体
        users = data['用户ID'].astype(str).unique()
        for user_id in users:
            self.graph.add_node(user_id, entity_type='User')
            self.entities['User'].add(user_id)
        print(f"✅ 抽取 {len(users)} 个用户实体")
        
        # 抽取品牌实体
        brands = data[['品牌', '品牌ID']].drop_duplicates()
        for _, row in brands.iterrows():
            brand_id = str(row['品牌ID'])
            self.graph.add_node(brand_id, entity_type='Brand', name=row['品牌'])
            self.entities['Brand'].add(brand_id)
        print(f"✅ 抽取 {len(brands)} 个品牌实体")
        
        # 抽取类别实体
        categories = data[['商品类别', '商品类目ID']].drop_duplicates()
        for _, row in categories.iterrows():
            category_id = str(row['商品类目ID'])
            self.graph.add_node(category_id, entity_type='Category', name=row['商品类别'])
            self.entities['Category'].add(category_id)
        print(f"✅ 抽取 {len(categories)} 个类别实体")
        
        # 抽取商品实体
        products = data[['商品ID', '商品名称', '售价', '价格区间']].drop_duplicates()
        for _, row in products.iterrows():
            product_id = str(row['商品ID'])
            self.graph.add_node(
                product_id, 
                entity_type='Product',
                name=row['商品名称'],
                price=row['售价'],
                price_range=row['价格区间']
            )
            self.entities['Product'].add(product_id)
        print(f"✅ 抽取 {len(products)} 个商品实体")
    
    def build_relations(self, data: pd.DataFrame) -> None:
        """构建实体关系"""
        print("正在构建实体关系...")
        
        # 构建商品-品牌关系
        product_brand_relations = data[['商品ID', '品牌ID']].drop_duplicates()
        for _, row in product_brand_relations.iterrows():
            product_id = str(row['商品ID'])
            brand_id = str(row['品牌ID'])
            self.graph.add_edge(product_id, brand_id, relation_type='BELONGS_TO_BRAND')
            self.graph.add_edge(brand_id, product_id, relation_type='PRODUCES')
        print(f"✅ 构建 {len(product_brand_relations)} 个商品-品牌关系")
        
        # 构建商品-类别关系
        product_category_relations = data[['商品ID', '商品类目ID']].drop_duplicates()
        for _, row in product_category_relations.iterrows():
            product_id = str(row['商品ID'])
            category_id = str(row['商品类目ID'])
            self.graph.add_edge(product_id, category_id, relation_type='BELONGS_TO_CATEGORY')
            self.graph.add_edge(category_id, product_id, relation_type='CONTAINS')
        print(f"✅ 构建 {len(product_category_relations)} 个商品-类别关系")
        
        # 构建用户行为关系
        behavior_relations = data[['用户ID', '商品ID', '行为类型', '时间戳']]
        for _, row in behavior_relations.iterrows():
            user_id = str(row['用户ID'])
            product_id = str(row['商品ID'])
            behavior_type = row['行为类型']
            timestamp = row['时间戳']
            
            # 使用唯一ID作为行为节点
            behavior_id = f"{user_id}_{product_id}_{timestamp}"
            self.graph.add_node(
                behavior_id, 
                entity_type='Behavior',
                type=behavior_type,
                timestamp=timestamp
            )
            
            # 构建用户-行为-商品的关系
            self.graph.add_edge(user_id, behavior_id, relation_type='PERFORMS')
            self.graph.add_edge(behavior_id, product_id, relation_type='TARGETS')
        print(f"✅ 构建 {len(behavior_relations)} 个用户行为关系")
    
    def build_graph(self, data_path: str) -> None:
        """构建完整的知识图谱"""
        print("=" * 50)
        print("开始构建电子模块知识图谱")
        print("=" * 50)
        
        # 先尝试从JSON文件加载数据
        if self.kg_json_path and os.path.exists(self.kg_json_path):
            self.load_from_json()
        
        # 加载CSV数据
        data = self.load_data(data_path)
        
        # 抽取实体
        self.extract_entities(data)
        
        # 构建关系
        self.build_relations(data)
        
        print("=" * 50)
        print(f"知识图谱构建完成")
        print(f"节点数量: {self.graph.number_of_nodes()}")
        print(f"边数量: {self.graph.number_of_edges()}")
        print(f"实体分布: {{'User': {len(self.entities['User'])}, 'Product': {len(self.entities['Product'])}, 'Brand': {len(self.entities['Brand'])}, 'Category': {len(self.entities['Category'])}, 'TechnicalParameter': {len(self.entities['TechnicalParameter'])}, 'UsageScenario': {len(self.entities['UsageScenario'])}, 'UserGroup': {len(self.entities['UserGroup'])}, 'Accessory': {len(self.entities['Accessory'])}}}")
        print("=" * 50)
    
    def save_graph(self, save_path: str = "models/electronic_knowledge_graph.pkl") -> None:
        """保存知识图谱到文件"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ 知识图谱已保存到 {save_path}")
    
    @staticmethod
    def load_graph(load_path: str = "models/electronic_knowledge_graph.pkl") -> "ElectronicKnowledgeGraph":
        """从文件加载知识图谱"""
        with open(load_path, 'rb') as f:
            kg = pickle.load(f)
        print(f"✅ 从 {load_path} 加载知识图谱")
        print(f"节点数量: {kg.graph.number_of_nodes()}")
        print(f"边数量: {kg.graph.number_of_edges()}")
        return kg
    
    def get_user_behavior(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户行为历史"""
        behaviors = []
        for neighbor in self.graph.neighbors(user_id):
            if self.graph.nodes[neighbor]['entity_type'] == 'Behavior':
                behavior_type = self.graph.nodes[neighbor]['type']
                timestamp = self.graph.nodes[neighbor]['timestamp']
                
                # 获取行为的目标商品
                for product in self.graph.neighbors(neighbor):
                    if self.graph.nodes[product]['entity_type'] == 'Product':
                        product_info = self.graph.nodes[product]
                        behaviors.append({
                            'product_id': product,
                            'product_name': product_info['name'],
                            'price': product_info['price'],
                            'behavior_type': behavior_type,
                            'timestamp': timestamp
                        })
        
        # 按时间戳排序
        return sorted(behaviors, key=lambda x: x['timestamp'], reverse=True)
    
    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """获取商品详细信息"""
        if product_id not in self.graph.nodes:
            return {}
        
        product_info = self.graph.nodes[product_id].copy()
        
        # 获取商品所属品牌
        for neighbor in self.graph.neighbors(product_id):
            if self.graph.nodes[neighbor]['entity_type'] == 'Brand':
                brand_info = self.graph.nodes[neighbor]
                product_info['brand'] = brand_info['name']
                product_info['brand_id'] = neighbor
                break
        
        # 获取商品所属类别
        for neighbor in self.graph.neighbors(product_id):
            if self.graph.nodes[neighbor]['entity_type'] == 'Category':
                category_info = self.graph.nodes[neighbor]
                product_info['category'] = category_info['name']
                product_info['category_id'] = neighbor
                break
        
        return product_info
    
    def get_related_products(self, product_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取相关商品（基于相同品牌或类别）"""
        if product_id not in self.graph.nodes:
            return []
        
        related_products = set()
        
        # 获取相同品牌的商品
        for neighbor in self.graph.neighbors(product_id):
            if self.graph.nodes[neighbor]['entity_type'] == 'Brand':
                for prod in self.graph.neighbors(neighbor):
                    if (self.graph.nodes[prod]['entity_type'] == 'Product' and 
                        prod != product_id):
                        related_products.add(prod)
        
        # 获取相同类别的商品
        for neighbor in self.graph.neighbors(product_id):
            if self.graph.nodes[neighbor]['entity_type'] == 'Category':
                for prod in self.graph.neighbors(neighbor):
                    if (self.graph.nodes[prod]['entity_type'] == 'Product' and 
                        prod != product_id):
                        related_products.add(prod)
        
        # 转换为推荐结果
        recommendations = []
        for prod in related_products:
            product_info = self.get_product_info(prod)
            recommendations.append({
                'product_id': prod,
                'name': product_info['name'],
                'brand': product_info.get('brand', 'unknown'),
                'category': product_info.get('category', 'unknown'),
                'price': product_info['price'],
                'price_range': product_info['price_range'],
                'score': 1.0  # 简单评分，实际可根据相似度调整
            })
        
        # 返回前top_k个结果
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def get_user_recommendations(self, user_id: str, top_k: int = 5, user_needs: str = '') -> List[Dict[str, Any]]:
        """基于用户行为和知识图谱生成推荐
        
        Args:
            user_id: 用户ID
            top_k: 返回推荐数量
            user_needs: 用户需求文本
        
        Returns:
            推荐商品列表
        """
        # 检查用户ID是否存在于图中
        if user_id not in self.graph.nodes:
            print(f"⚠️ 用户 {user_id} 不存在于知识图谱中，返回热门商品推荐")
            return self.get_popular_products(top_k)
        
        # 获取用户行为
        user_behaviors = self.get_user_behavior(user_id)
        
        # 存储所有推荐结果
        all_recommendations = []
        seen_product_ids = set()
        
        # 1. 基于用户最近浏览的商品推荐相关商品
        if user_behaviors:
            recent_product = user_behaviors[0]['product_id']
            related_products = self.get_related_products(recent_product, top_k)
            for product in related_products:
                if product['product_id'] not in seen_product_ids:
                    seen_product_ids.add(product['product_id'])
                    all_recommendations.append(product)
        
        # 2. 基于用户需求进行推荐
        if user_needs:
            # 提取用户需求中的关键词
            user_keywords = user_needs.lower().split()
            
            # 遍历所有商品
            for node in self.graph.nodes:
                if self.graph.nodes[node]['entity_type'] == 'Product':
                    product_info = self.get_product_info(node)
                    if not product_info:
                        continue
                    
                    # 计算匹配分数
                    match_score = 0.0
                    
                    # 检查品牌匹配
                    if 'brand' in product_info:
                        brand = product_info['brand'].lower()
                        if any(keyword in brand for keyword in user_keywords):
                            match_score += 0.3
                    
                    # 检查类别匹配
                    if 'category' in product_info:
                        category = product_info['category'].lower()
                        if any(keyword in category for keyword in user_keywords):
                            match_score += 0.3
                    
                    # 检查商品名称匹配
                    if 'name' in product_info:
                        product_name = product_info['name'].lower()
                        if any(keyword in product_name for keyword in user_keywords):
                            match_score += 0.4
                    
                    # 如果匹配分数大于0，添加到推荐列表
                    if match_score > 0 and node not in seen_product_ids:
                        seen_product_ids.add(node)
                        recommendation = {
                            'product_id': node,
                            'name': product_info['name'],
                            'brand': product_info.get('brand', 'unknown'),
                            'category': product_info.get('category', 'unknown'),
                            'price': product_info['price'],
                            'price_range': product_info.get('price_range', ''),
                            'score': match_score
                        }
                        all_recommendations.append(recommendation)
        
        # 3. 如果推荐数量不足，添加热门商品
        if len(all_recommendations) < top_k:
            popular_products = self.get_popular_products(top_k - len(all_recommendations))
            for product in popular_products:
                if product['product_id'] not in seen_product_ids:
                    seen_product_ids.add(product['product_id'])
                    all_recommendations.append(product)
        
        # 4. 排序并返回
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return all_recommendations[:top_k]
    
    def get_popular_products(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取热门商品（基于行为次数）"""
        # 统计每个商品的行为次数
        product_behavior_count = {}
        
        for node in self.graph.nodes:
            if self.graph.nodes[node]['entity_type'] == 'Product':
                behavior_count = 0
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['entity_type'] == 'Behavior':
                        behavior_count += 1
                product_behavior_count[node] = behavior_count
        
        # 按行为次数排序
        sorted_products = sorted(product_behavior_count.items(), key=lambda x: x[1], reverse=True)
        
        # 转换为推荐结果
        recommendations = []
        for product_id, count in sorted_products[:top_k]:
            product_info = self.get_product_info(product_id)
            recommendations.append({
                'product_id': product_id,
                'name': product_info['name'],
                'brand': product_info.get('brand', 'unknown'),
                'category': product_info.get('category', 'unknown'),
                'price': product_info['price'],
                'price_range': product_info['price_range'],
                'score': count  # 使用行为次数作为评分
            })
        
        return recommendations

# 测试代码
if __name__ == "__main__":
    # 构建知识图谱
    # 1. 从JSON文件加载知识图谱数据（如果存在）
    kg_json_path = os.path.join("data", "电子数码", "knowledge_graph.json")
    kg = ElectronicKnowledgeGraph(kg_json_path=kg_json_path)
    
    # 2. 从CSV文件加载数据并构建完整知识图谱
    data_path = 'data/电子产品用户行为数据.csv'
    if os.path.exists(data_path):
        kg.build_graph(data_path)
    else:
        print(f"⚠️  数据文件 {data_path} 不存在，仅使用JSON数据构建知识图谱")
    
    # 3. 保存知识图谱
    kg.save_graph()
    
    # 测试知识图谱功能
    print("\n=== 测试知识图谱功能 ===")
    
    # 测试获取用户行为
    if kg.entities['User']:
        sample_user = list(kg.entities['User'])[0]
        user_behaviors = kg.get_user_behavior(sample_user)
        print(f"用户 {sample_user} 的最近行为: {user_behaviors[:2]}")
    
    # 测试获取商品信息
    if kg.entities['Product']:
        sample_product = list(kg.entities['Product'])[0]
        product_info = kg.get_product_info(sample_product)
        print(f"商品 {sample_product} 的信息: {product_info}")
        
        # 测试获取相关商品
        related_products = kg.get_related_products(sample_product, 3)
        print(f"商品 {sample_product} 的相关商品: {related_products}")
        
        # 测试获取用户推荐（带用户需求）
        if kg.entities['User']:
            user_recommendations = kg.get_user_recommendations(sample_user, 3, "我需要一台高性能的笔记本电脑")
            print(f"给用户 {sample_user} 的推荐: {user_recommendations}")
    
    # 测试获取热门商品
    popular_products = kg.get_popular_products(3)
    print(f"热门商品: {popular_products}")
