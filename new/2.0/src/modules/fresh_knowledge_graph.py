#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品知识图谱构建实现
从商品数据中抽取实体和属性，构建知识图谱
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class KnowledgeGraphBuilder:
    """商品知识图谱构建类"""
    
    def __init__(self, data_path="goods_info.csv"):
        """初始化知识图谱构建器
        
        Args:
            data_path: 商品数据文件路径
        """
        self.data_path = data_path
        self.goods_df = None
        self.graph = nx.Graph()
        self.entities = {
            '商品': set(),
            '商家': set(),
            '品类': set()
        }
    
    def load_data(self):
        """加载商品数据"""
        print("加载商品数据...")
        self.goods_df = pd.read_csv(self.data_path)
        print(f"✅ 加载完成，共 {len(self.goods_df)} 个商品")
    
    def extract_entities(self):
        """抽取实体"""
        if self.goods_df is None:
            self.load_data()
        
        print("开始抽取实体...")
        
        # 抽取商品实体
        for _, row in self.goods_df.iterrows():
            self.entities['商品'].add(row['名称'])
            self.entities['商家'].add(row['商家'])
            self.entities['品类'].add(row['品类'])
        
        print(f"✅ 实体抽取完成")
        print(f"  - 商品实体：{len(self.entities['商品'])}个")
        print(f"  - 商家实体：{len(self.entities['商家'])}个")
        print(f"  - 品类实体：{len(self.entities['品类'])}个")
    
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
            self.graph.add_edge(row['名称'], row['品类'], relation="属于", type="品类关系")
            
            # 商品-商家关系
            self.graph.add_edge(row['名称'], row['商家'], relation="由...销售", type="销售关系")
            
            # 商品属性
            self.graph.nodes[row['名称']]['单价'] = row['单价']
            self.graph.nodes[row['名称']]['评价'] = row['评价']
            self.graph.nodes[row['名称']]['ID'] = row['ID']
        
        print(f"✅ 知识图谱构建完成")
        print(f"  - 节点数量：{self.graph.number_of_nodes()}")
        print(f"  - 边数量：{self.graph.number_of_edges()}")
    
    def save_graph(self, save_path="knowledge_graph.pkl"):
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
    
    def load_graph(self, load_path="knowledge_graph.pkl"):
        """从文件加载知识图谱
        
        Args:
            load_path: 加载文件路径
        """
        print(f"从 {load_path} 加载知识图谱...")
        with open(load_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✅ 知识图谱加载完成")
        print(f"  - 节点数量：{self.graph.number_of_nodes()}")
        print(f"  - 边数量：{self.graph.number_of_edges()}")
    
    def visualize_graph(self, max_nodes=50, figsize=(15, 15)):
        """可视化知识图谱
        
        Args:
            max_nodes: 最大显示节点数量
            figsize: 图形大小
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        print("可视化知识图谱...")
        
        # 限制显示的节点数量，避免图形过于复杂
        if self.graph.number_of_nodes() > max_nodes:
            # 选择部分节点进行可视化（按度数排序）
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph(top_nodes)
        else:
            subgraph = self.graph
        
        # 设置节点颜色
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', '未知')
            if node_type == '商品':
                node_colors.append('#1f77b4')  # 蓝色
            elif node_type == '商家':
                node_colors.append('#2ca02c')  # 绿色
            elif node_type == '品类':
                node_colors.append('#ff7f0e')  # 橙色
            else:
                node_colors.append('#d62728')  # 红色
        
        # 绘制图形
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=0.3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            subgraph, pos, node_size=500, node_color=node_colors, alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            subgraph, pos, width=1.0, alpha=0.5
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            subgraph, pos, font_size=10, font_family='SimHei'
        )
        
        # 绘制边标签
        edge_labels = nx.get_edge_attributes(subgraph, 'relation')
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_size=8, font_family='SimHei'
        )
        
        # 设置标题
        plt.title('商品知识图谱可视化', fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存可视化结果
        plt.savefig('knowledge_graph_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ 知识图谱可视化完成，已保存为 knowledge_graph_visualization.png")
        
    def get_related_entities(self, entity_name, relation_type=None):
        """获取相关实体
        
        Args:
            entity_name: 实体名称
            relation_type: 关系类型过滤
            
        Returns:
            相关实体列表
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
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
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        if category_name not in self.graph.nodes():
            print(f"❌ 品类 {category_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        for neighbor in self.graph.neighbors(category_name):
            if self.graph.nodes[neighbor]['type'] == '商品':
                goods_list.append({
                    '商品名称': neighbor,
                    '单价': self.graph.nodes[neighbor].get('单价', '未知'),
                    '评价': self.graph.nodes[neighbor].get('评价', '未知')
                })
        
        return goods_list
    
    def query_goods_by_merchant(self, merchant_name):
        """根据商家查询商品
        
        Args:
            merchant_name: 商家名称
            
        Returns:
            商品列表
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        if merchant_name not in self.graph.nodes():
            print(f"❌ 商家 {merchant_name} 不存在于知识图谱中")
            return []
        
        goods_list = []
        for neighbor in self.graph.neighbors(merchant_name):
            if self.graph.nodes[neighbor]['type'] == '商品':
                goods_list.append({
                    '商品名称': neighbor,
                    '单价': self.graph.nodes[neighbor].get('单价', '未知'),
                    '评价': self.graph.nodes[neighbor].get('评价', '未知')
                })
        
        return goods_list
    
    def entity_linking(self, text):
        """实体链接：将文本中的关键词链接到知识图谱中的实体
        
        Args:
            text: 用户输入文本
            
        Returns:
            链接到的实体列表
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
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
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
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
    
    def multi_hop_query(self, start_entity, relation_type, hops=2):
        """多跳关系查询
        
        Args:
            start_entity: 起始实体
            relation_type: 关系类型
            hops: 跳数
            
        Returns:
            多跳查询结果
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        if start_entity not in self.graph.nodes():
            print(f"❌ 实体 {start_entity} 不存在于知识图谱中")
            return []
        
        visited = set()
        results = []
        
        def dfs(entity, current_hop, path):
            if current_hop > hops:
                return
            
            visited.add(entity)
            
            for neighbor in self.graph.neighbors(entity):
                edge_data = self.graph.get_edge_data(entity, neighbor)
                if edge_data['type'] == relation_type:
                    new_path = path + [{
                        '实体': neighbor,
                        '类型': self.graph.nodes[neighbor]['type'],
                        '关系': edge_data['relation']
                    }]
                    results.append({
                        '路径': new_path,
                        '跳数': current_hop
                    })
                    if current_hop < hops:
                        dfs(neighbor, current_hop + 1, new_path)
        
        dfs(start_entity, 1, [{
            '实体': start_entity,
            '类型': self.graph.nodes[start_entity]['type'],
            '关系': '起点'
        }])
        
        return results
    
    def kg_qa(self, question):
        """知识图谱问答
        
        Args:
            question: 用户问题
            
        Returns:
            问答结果
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        # 简单的问答逻辑示例
        # 实际应用中可以使用更复杂的NLP技术
        
        # 提取实体
        entities = self.entity_linking(question)
        if not entities:
            return "抱歉，我无法理解您的问题。"
        
        # 根据问题类型处理
        if '什么' in question or '哪些' in question:
            # 查询类问题
            for entity in entities:
                if entity['实体类型'] == '品类':
                    goods = self.query_goods_by_category(entity['实体名称'])
                    if goods:
                        return f"{entity['实体名称']}类别包含以下商品：{', '.join([g['商品名称'] for g in goods[:10]])}等。"
        
        elif '哪里' in question or '哪个' in question:
            # 地点/商家类问题
            for entity in entities:
                if entity['实体类型'] == '商品':
                    relations = self.get_entity_relations(entity['实体名称'])
                    merchants = [r['目标实体'] for r in relations if r['目标实体类型'] == '商家']
                    if merchants:
                        return f"{entity['实体名称']}可以在以下商家购买：{', '.join(merchants[:5])}等。"
        
        return "抱歉，我无法回答这个问题。"

if __name__ == "__main__":
    # 示例用法
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.load_data()
    kg_builder.extract_entities()
    kg_builder.build_graph()
    kg_builder.save_graph()
    
    # 可视化知识图谱
    kg_builder.visualize_graph(max_nodes=30)
    
    # 查询示例
    print("\n=== 查询示例 ===")
    
    # 查询乳制品类别下的商品
    print("1. 乳制品类别下的商品：")
    dairy_goods = kg_builder.query_goods_by_category("乳制品")
    for i, goods in enumerate(dairy_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 单价：{goods['单价']} - 评价：{goods['评价']}")
    
    # 查询朴朴超市销售的商品
    print("\n2. 朴朴超市销售的商品：")
    popo_goods = kg_builder.query_goods_by_merchant("朴朴超市（石景山店）")
    for i, goods in enumerate(popo_goods[:3], 1):
        print(f"   {i}. {goods['商品名称']} - 单价：{goods['单价']} - 评价：{goods['评价']}")
    
    # 查询商品"鲜牛奶"的相关实体
    print("\n3. 商品'鲜牛奶'的相关实体：")
    milk_related = kg_builder.get_related_entities("鲜牛奶")
    for i, entity in enumerate(milk_related, 1):
        print(f"   {i}. {entity['实体名称']}（{entity['实体类型']}） - {entity['关系']}")
