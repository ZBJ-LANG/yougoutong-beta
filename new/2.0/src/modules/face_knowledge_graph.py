#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美妆护肤知识图谱构建实现
从商品数据中抽取实体和属性，构建知识图谱
"""

import pandas as pd
import networkx as nx
import pickle
import os
import json

class FaceKnowledgeGraph:
    """美妆护肤知识图谱构建类"""
    
    def __init__(self, data_path=None, kg_json_path=None):
        """初始化知识图谱构建器
        
        Args:
            data_path: 商品数据文件路径
            kg_json_path: 知识图谱JSON文件路径
        """
        self.data_path = data_path
        self.kg_json_path = kg_json_path
        self.goods_df = None
        self.graph = nx.DiGraph()
        self.entities = {
            '商品': set(),
            '品牌': set(),
            '品类': set(),
            '成分': set(),
            '功效': set(),
            '肤质': set()
        }
    
    def load_data(self):
        """加载商品数据"""
        print("加载美妆护肤商品数据...")
        self.goods_df = pd.read_csv(self.data_path)
        print(f"✅ 加载完成，共 {len(self.goods_df)} 个商品")
    
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
                        self.entities['品牌'].add(entity['name'])
                elif entity_type == 'Product':
                    for entity in entities:
                        self.entities['商品'].add(entity['name'])
                elif entity_type == 'Ingredient':
                    for entity in entities:
                        self.entities['成分'].add(entity['name'])
                elif entity_type == 'Efficacy':
                    for entity in entities:
                        self.entities['功效'].add(entity['name'])
                elif entity_type == 'SkinType':
                    for entity in entities:
                        self.entities['肤质'].add(entity['name'])
            
            # 加载关系
            for relation in kg_data.get('relations', []):
                source = relation['source']
                target = relation['target']
                rel_type = relation['type']
                
                # 添加节点
                if source not in self.graph.nodes:
                    # 推断节点类型
                    node_type = self._infer_entity_type(source)
                    self.graph.add_node(source, type=node_type)
                if target not in self.graph.nodes:
                    node_type = self._infer_entity_type(target)
                    self.graph.add_node(target, type=node_type)
                
                # 添加边
                relation_map = {
                    'has_ingredient': {'relation': '包含', 'type': '成分关系'},
                    'has_efficacy': {'relation': '具有', 'type': '功效关系'},
                    'suitable_for': {'relation': '适合', 'type': '肤质关系'},
                    'belongs_to': {'relation': '属于', 'type': '品牌关系'},
                    'category': {'relation': '属于', 'type': '品类关系'}
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
    
    def extract_entities(self):
        """抽取实体"""
        if self.goods_df is None:
            self.load_data()
        
        print("开始抽取实体...")
        
        # 抽取商品实体
        for _, row in self.goods_df.iterrows():
            self.entities['商品'].add(row['名称'])
            # 使用商家作为品牌
            self.entities['品牌'].add(row['商家'])
            self.entities['品类'].add(row['品类'])
            
            # 如果有成分信息，抽取成分实体
            if '成分' in self.goods_df.columns and pd.notna(row['成分']):
                ingredients = row['成分'].split(',')
                for ingredient in ingredients:
                    self.entities['成分'].add(ingredient.strip())
        
        print(f"✅ 实体抽取完成")
        print(f"  - 商品实体：{len(self.entities['商品'])}个")
        print(f"  - 品牌实体：{len(self.entities['品牌'])}个")
        print(f"  - 品类实体：{len(self.entities['品类'])}个")
        if '成分' in self.entities:
            print(f"  - 成分实体：{len(self.entities['成分'])}个")
    
    def build_graph(self):
        """构建知识图谱"""
        if not self.entities['商品']:
            self.extract_entities()
        
        print("开始构建知识图谱...")
        
        # 添加实体节点
        for entity_type, entities in self.entities.items():
            for entity in entities:
                self.graph.add_node(entity, type=entity_type)
        
        # 添加关系边
        if self.goods_df is not None:
            for _, row in self.goods_df.iterrows():
                # 商品-品类关系
                self.graph.add_edge(row['名称'], row['品类'], relation="属于", type="品类关系")
                
                # 商品-品牌关系（使用商家作为品牌）
                self.graph.add_edge(row['名称'], row['商家'], relation="由...销售", type="品牌关系")
                
                # 商品属性
                self.graph.nodes[row['名称']]['单价'] = row['单价']
                self.graph.nodes[row['名称']]['评价'] = row['评价']
                self.graph.nodes[row['名称']]['ID'] = row['ID']
                
                # 如果有成分信息，添加商品-成分关系
                if '成分' in self.goods_df.columns and pd.notna(row['成分']):
                    ingredients = row['成分'].split(',')
                    for ingredient in ingredients:
                        ingredient = ingredient.strip()
                        self.graph.add_edge(row['名称'], ingredient, relation="包含", type="成分关系")
                
                # 如果有功效信息，添加商品-功效关系
                if '功效' in self.goods_df.columns and pd.notna(row['功效']):
                    efficacies = row['功效'].split(',')
                    for efficacy in efficacies:
                        efficacy = efficacy.strip()
                        self.graph.add_edge(row['名称'], efficacy, relation="具有", type="功效关系")
                
                # 如果有肤质信息，添加商品-肤质关系
                if '肤质' in self.goods_df.columns and pd.notna(row['肤质']):
                    skin_types = row['肤质'].split(',')
                    for skin_type in skin_types:
                        skin_type = skin_type.strip()
                        self.graph.add_edge(row['名称'], skin_type, relation="适合", type="肤质关系")
        
        print(f"✅ 知识图谱构建完成")
        print(f"  - 节点数量：{self.graph.number_of_nodes()}")
        print(f"  - 边数量：{self.graph.number_of_edges()}")
    
    def save_graph(self, save_path="face_knowledge_graph.pkl"):
        """保存知识图谱到文件
        
        Args:
            save_path: 保存文件路径
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        print(f"保存知识图谱到 {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print("✅ 知识图谱保存完成")
    
    @classmethod
    def load_graph(cls, load_path="face_knowledge_graph.pkl"):
        """从文件加载知识图谱
        
        Args:
            load_path: 加载文件路径
        
        Returns:
            FaceKnowledgeGraph: 知识图谱实例
        """
        print(f"从 {load_path} 加载知识图谱...")
        kg_instance = cls()
        with open(load_path, 'rb') as f:
            kg_instance.graph = pickle.load(f)
        print(f"✅ 知识图谱加载完成")
        print(f"  - 节点数量：{kg_instance.graph.number_of_nodes()}")
        print(f"  - 边数量：{kg_instance.graph.number_of_edges()}")
        return kg_instance
    
    def get_related_entities(self, entity_name, relation_type=None):
        """获取相关实体
        
        Args:
            entity_name: 实体名称
            relation_type: 关系类型过滤
            
        Returns:
            相关实体列表
        """
        if entity_name not in self.graph.nodes():
            print(f"❌ 实体 {entity_name} 不存在于知识图谱中")
            return []
        
        related_entities = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.get_edge_data(entity_name, neighbor)
            if not relation_type or edge_data['type'] == relation_type:
                related_entities.append({
                    '实体名称': neighbor,
                    '实体类型': self.graph.nodes[neighbor]['type'],
                    '关系': edge_data['relation'],
                    '关系类型': edge_data['type']
                })
        
        return related_entities
    
    def query_goods_by_category(self, category_name):
        """根据品类查询商品
        
        Args:
            category_name: 品类名称
            
        Returns:
            商品列表
        """
        if category_name not in self.graph.nodes():
            print(f"❌ 品类 {category_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        # 查找所有指向该品类的商品节点
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                if category_name in self.graph.neighbors(node):
                    goods_list.append({
                        '商品名称': node,
                        '单价': self.graph.nodes[node].get('单价', '未知'),
                        '评价': self.graph.nodes[node].get('评价', '未知')
                    })
        
        return goods_list
    
    def query_goods_by_brand(self, brand_name):
        """根据品牌查询商品
        
        Args:
            brand_name: 品牌名称
            
        Returns:
            商品列表
        """
        if brand_name not in self.graph.nodes():
            print(f"❌ 品牌 {brand_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        # 查找所有指向该品牌的商品节点
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                if brand_name in self.graph.neighbors(node):
                    goods_list.append({
                        '商品名称': node,
                        '单价': self.graph.nodes[node].get('单价', '未知'),
                        '评价': self.graph.nodes[node].get('评价', '未知')
                    })
        
        return goods_list
    
    def entity_linking(self, text):
        """实体链接：将文本中的关键词链接到知识图谱中的实体
        
        Args:
            text: 用户输入文本
            
        Returns:
            链接到的实体列表
        """
        linked_entities = []
        
        # 遍历所有实体，查找文本中包含的实体
        for node in self.graph.nodes():
            if node in text:
                linked_entities.append({
                    '实体名称': node,
                    '实体类型': self.graph.nodes[node]['type'],
                    '匹配文本': node
                })
        
        # 按匹配长度排序，优先返回长匹配
        linked_entities.sort(key=lambda x: len(x['匹配文本']), reverse=True)
        
        return linked_entities
    
    def get_entity_relations(self, entity_name):
        """获取实体的所有关系
        
        Args:
            entity_name: 实体名称
            
        Returns:
            实体关系列表
        """
        if entity_name not in self.graph.nodes():
            print(f"❌ 实体 {entity_name} 不存在于知识图谱中")
            return []
        
        relations = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.get_edge_data(entity_name, neighbor)
            relations.append({
                '目标实体': neighbor,
                '目标实体类型': self.graph.nodes[neighbor]['type'],
                '关系': edge_data['relation'],
                '关系类型': edge_data['type']
            })
        
        return relations
    
    def get_user_recommendations(self, user_id, top_k=10, user_needs=None):
        """获取用户推荐
        
        Args:
            user_id: 用户ID
            top_k: 返回推荐数量
            user_needs: 用户需求文本
            
        Returns:
            推荐商品列表
        """
        recommendations = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                # 基础评分：使用评价分数
                try:
                    base_rating = float(self.graph.nodes[node].get('评价', '0').replace('分', ''))
                except (ValueError, AttributeError):
                    base_rating = 0.0
                
                # 计算额外分数
                extra_score = 0.0
                
                # 1. 如果有用户需求，进行需求匹配
                if user_needs:
                    # 提取商品相关的实体
                    related_entities = set()
                    for neighbor in self.graph.neighbors(node):
                        related_entities.add(neighbor)
                        related_entities.add(self.graph.nodes[neighbor].get('type', ''))
                    
                    # 计算需求匹配度
                    user_needs_lower = user_needs.lower()
                    match_count = sum(1 for entity in related_entities if str(entity).lower() in user_needs_lower)
                    extra_score += match_count * 0.5
                
                # 2. 功效匹配加分
                efficacy_count = len([n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '功效'])
                extra_score += efficacy_count * 0.1
                
                # 3. 成分丰富度加分
                ingredient_count = len([n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '成分'])
                extra_score += ingredient_count * 0.05
                
                # 总评分
                total_score = base_rating + extra_score
                
                recommendations.append({
                    'product_id': self.graph.nodes[node].get('ID', node),
                    'brand': next((n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '品牌'), '未知'),
                    'category': next((n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '品类'), '未知'),
                    'price': self.graph.nodes[node].get('单价', 0.0),
                    'score': total_score,
                    'name': node
                })
        
        # 按评分排序，返回top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]

if __name__ == "__main__":
    # 示例用法
    # 1. 从JSON文件加载知识图谱
    kg_json_path = os.path.join("data", "face_data", "美妆护肤", "knowledge_graph.json")
    kg_builder = FaceKnowledgeGraph(kg_json_path=kg_json_path)
    kg_builder.load_from_json()
    
    # 2. 同时从CSV加载商品数据（可选）
    data_path = os.path.join("data", "face_data", "face_goods_info.csv")
    if os.path.exists(data_path):
        kg_builder.data_path = data_path
        kg_builder.load_data()
        kg_builder.extract_entities()
        kg_builder.build_graph()
    
    # 保存知识图谱
    save_path = os.path.join("models", "face_knowledge_graph.pkl")
    kg_builder.save_graph(save_path)
    
    # 查询示例
    print("\n=== 查询示例 ===")
    
    # 查询某个品类下的商品
    category = "面霜"
    print(f"1. {category}类别下的商品：")
    category_goods = kg_builder.query_goods_by_category(category)
    for i, goods in enumerate(category_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 单价：{goods['单价']} - 评价：{goods['评价']}")
    
    # 查询某个品牌下的商品
    brand = "雅诗兰黛"
    print(f"\n2. {brand}品牌下的商品：")
    brand_goods = kg_builder.query_goods_by_brand(brand)
    for i, goods in enumerate(brand_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 单价：{goods['单价']} - 评价：{goods['评价']}")
    
    # 测试用户推荐
    print(f"\n3. 测试用户推荐：")
    recommendations = kg_builder.get_user_recommendations('user_1', top_k=5, user_needs='我需要一款补水保湿的面霜')
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['brand']} {rec['category']} - 价格：{rec['price']} - 推荐分数：{rec['score']:.4f}")
