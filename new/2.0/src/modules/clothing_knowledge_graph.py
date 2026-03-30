#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服装穿搭知识图谱构建实现
从商品数据中抽取实体和属性，构建知识图谱
"""

import pandas as pd
import networkx as nx
import pickle
import os

class ClothingKnowledgeGraph:
    """服装穿搭知识图谱构建类"""
    
    def __init__(self, data_path=None):
        """初始化知识图谱构建器
        
        Args:
            data_path: 商品数据文件路径
        """
        self.data_path = data_path
        self.goods_df = None
        self.graph = nx.DiGraph()
        self.entities = {
            '商品': set(),
            '品牌': set(),
            '品类': set(),
            '风格': set(),
            '季节': set()
        }
    
    def load_data(self):
        """加载商品数据"""
        print("加载服装穿搭商品数据...")
        self.goods_df = pd.read_csv(self.data_path)
        print(f"✅ 加载完成，共 {len(self.goods_df)} 个商品")
    
    def extract_entities(self):
        """抽取实体"""
        if self.goods_df is None:
            self.load_data()
        
        print("开始抽取实体...")
        
        # 抽取商品实体
        for _, row in self.goods_df.iterrows():
            self.entities['商品'].add(row['商品名称'])
            self.entities['品牌'].add(row['品牌'])
            self.entities['品类'].add(row['商品类别'])
            
            # 抽取风格实体
            if '风格标签' in self.goods_df.columns and pd.notna(row['风格标签']):
                self.entities['风格'].add(row['风格标签'])
            
            # 抽取季节实体
            if '季节标签' in self.goods_df.columns and pd.notna(row['季节标签']):
                self.entities['季节'].add(row['季节标签'])
        
        print(f"✅ 实体抽取完成")
        print(f"  - 商品实体：{len(self.entities['商品'])}个")
        print(f"  - 品牌实体：{len(self.entities['品牌'])}个")
        print(f"  - 品类实体：{len(self.entities['品类'])}个")
        if '风格' in self.entities:
            print(f"  - 风格实体：{len(self.entities['风格'])}个")
        if '季节' in self.entities:
            print(f"  - 季节实体：{len(self.entities['季节'])}个")
    
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
        for _, row in self.goods_df.iterrows():
            # 商品-品类关系
            self.graph.add_edge(row['商品名称'], row['商品类别'], relation="属于", type="品类关系")
            
            # 商品-品牌关系
            self.graph.add_edge(row['商品名称'], row['品牌'], relation="由...生产", type="品牌关系")
            
            # 商品-风格关系
            if '风格标签' in self.goods_df.columns and pd.notna(row['风格标签']):
                self.graph.add_edge(row['商品名称'], row['风格标签'], relation="具有...风格", type="风格关系")
            
            # 商品-季节关系
            if '季节标签' in self.goods_df.columns and pd.notna(row['季节标签']):
                self.graph.add_edge(row['商品名称'], row['季节标签'], relation="适合...季节", type="季节关系")
            
            # 商品属性
            self.graph.nodes[row['商品名称']]['售价'] = row['售价']
            self.graph.nodes[row['商品名称']]['商品ID'] = row['商品ID']
        
        print(f"✅ 知识图谱构建完成")
        print(f"  - 节点数量：{self.graph.number_of_nodes()}")
        print(f"  - 边数量：{self.graph.number_of_edges()}")
    
    def save_graph(self, save_path="clothing_knowledge_graph.pkl"):
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
    def load_graph(cls, load_path="clothing_knowledge_graph.pkl"):
        """从文件加载知识图谱
        
        Args:
            load_path: 加载文件路径
        
        Returns:
            ClothingKnowledgeGraph: 知识图谱实例
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
                        '售价': self.graph.nodes[node].get('售价', '未知'),
                        '商品ID': self.graph.nodes[node].get('商品ID', '未知')
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
                        '售价': self.graph.nodes[node].get('售价', '未知'),
                        '商品ID': self.graph.nodes[node].get('商品ID', '未知')
                    })
        
        return goods_list
    
    def query_goods_by_style(self, style_name):
        """根据风格查询商品
        
        Args:
            style_name: 风格名称
            
        Returns:
            商品列表
        """
        if style_name not in self.graph.nodes():
            print(f"❌ 风格 {style_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        # 查找所有指向该风格的商品节点
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                if style_name in self.graph.neighbors(node):
                    goods_list.append({
                        '商品名称': node,
                        '售价': self.graph.nodes[node].get('售价', '未知'),
                        '商品ID': self.graph.nodes[node].get('商品ID', '未知')
                    })
        
        return goods_list
    
    def query_goods_by_season(self, season_name):
        """根据季节查询商品
        
        Args:
            season_name: 季节名称
            
        Returns:
            商品列表
        """
        if season_name not in self.graph.nodes():
            print(f"❌ 季节 {season_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        # 查找所有指向该季节的商品节点
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                if season_name in self.graph.neighbors(node):
                    goods_list.append({
                        '商品名称': node,
                        '售价': self.graph.nodes[node].get('售价', '未知'),
                        '商品ID': self.graph.nodes[node].get('商品ID', '未知')
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
    
    def get_user_recommendations(self, user_id, top_k=10):
        """获取用户推荐（简化实现）
        
        Args:
            user_id: 用户ID
            top_k: 返回推荐数量
            
        Returns:
            推荐商品列表
        """
        # 简化实现：返回评分最高的商品
        recommendations = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == '商品':
                # 简化处理，使用随机分数作为推荐分数
                import random
                score = random.uniform(0.5, 1.0)
                
                # 获取品牌
                brand = next((n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '品牌'), '未知')
                
                # 获取品类
                category = next((n for n in self.graph.neighbors(node) if self.graph.nodes[n]['type'] == '品类'), '未知')
                
                recommendations.append({
                    'product_id': self.graph.nodes[node].get('商品ID', node),
                    'brand': brand,
                    'category': category,
                    'price': self.graph.nodes[node].get('售价', 0.0),
                    'score': score
                })
        
        # 按评分排序，返回top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]

if __name__ == "__main__":
    # 示例用法
    data_path = os.path.join("data", "服装穿搭", "服装穿搭用户行为数据.csv")
    kg_builder = ClothingKnowledgeGraph(data_path)
    kg_builder.load_data()
    kg_builder.extract_entities()
    kg_builder.build_graph()
    
    # 保存知识图谱
    save_path = os.path.join("models", "clothing_knowledge_graph.pkl")
    kg_builder.save_graph(save_path)
    
    # 查询示例
    print("\n=== 查询示例 ===")
    
    # 查询某个品类下的商品
    category = "运动裤"
    print(f"1. {category}类别下的商品：")
    category_goods = kg_builder.query_goods_by_category(category)
    for i, goods in enumerate(category_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 价格：{goods['售价']}元")
    
    # 查询某个风格下的商品
    style = "美式"
    print(f"\n2. {style}风格下的商品：")
    style_goods = kg_builder.query_goods_by_style(style)
    for i, goods in enumerate(style_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 价格：{goods['售价']}元")
    
    # 查询某个季节下的商品
    season = "冬"
    print(f"\n3. {season}季适合的商品：")
    season_goods = kg_builder.query_goods_by_season(season)
    for i, goods in enumerate(season_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 价格：{goods['售价']}元")
