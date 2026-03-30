#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品语义向量化存储实现
使用阿里云text-embedding-v1模型生成语义向量
"""

import os
import pandas as pd
import dashscope
from dashscope import TextEmbedding
import numpy as np
import pickle
from dotenv import load_dotenv

# 从.env文件加载配置
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path, override=True)

# 从环境变量获取API密钥
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("未配置DASHSCOPE_API_KEY环境变量")

dashscope.api_key = API_KEY
print(f"已从{env_path}加载API密钥：{API_KEY[:5]}...{API_KEY[-5:]}")

class SemanticVectorStore:
    """商品语义向量存储类"""
    
    def __init__(self, data_path="goods_info.csv", embedding_model="text-embedding-v1"):
        """初始化向量存储
        
        Args:
            data_path: 商品数据文件路径
            embedding_model: 嵌入模型名称
        """
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.goods_df = None
        self.embeddings = None
        self.vector_store = {}
    
    def load_data(self):
        """加载商品数据"""
        print("加载商品数据...")
        self.goods_df = pd.read_csv(self.data_path)
        print(f"✅ 加载完成，共 {len(self.goods_df)} 个商品")
    
    def generate_embeddings(self, batch_size=10):
        """生成商品语义向量
        
        Args:
            batch_size: 批量处理大小
        """
        if self.goods_df is None:
            self.load_data()
        
        print("开始生成商品语义向量...")
        
        # 批量生成向量
        total_batches = (len(self.goods_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(self.goods_df), batch_size):
            batch_goods = self.goods_df.iloc[i:i+batch_size]
            batch_ids = batch_goods['ID'].tolist()
            
            # 生成多种向量表示
            # 1. 商品名称向量
            name_texts = [row['名称'] for _, row in batch_goods.iterrows()]
            # 2. 商品完整描述向量
            desc_texts = [f"商品名称：{row['名称']}，商家：{row['商家']}，单价：{row['单价']}，评价：{row['评价']}，品类：{row['品类']}" for _, row in batch_goods.iterrows()]
            # 3. 商品属性向量
            attr_texts = [f"品类：{row['品类']}，单价：{row['单价']}" for _, row in batch_goods.iterrows()]
            
            try:
                # 调用阿里云text-embedding-v1模型生成名称向量
                name_resp = TextEmbedding.call(
                    model=self.embedding_model,
                    input=name_texts
                )
                
                # 生成描述向量
                desc_resp = TextEmbedding.call(
                    model=self.embedding_model,
                    input=desc_texts
                )
                
                # 生成属性向量
                attr_resp = TextEmbedding.call(
                    model=self.embedding_model,
                    input=attr_texts
                )
                
                if (name_resp.status_code == 200 and 
                    desc_resp.status_code == 200 and 
                    attr_resp.status_code == 200):
                    
                    # 处理名称向量
                    if isinstance(name_resp, dict) and 'output' in name_resp:
                        name_embeddings = [item['embedding'] for item in name_resp['output']['embeddings']]
                    elif hasattr(name_resp, 'output'):
                        name_embeddings = [item.embedding for item in name_resp.output.embeddings]
                    
                    # 处理描述向量
                    if isinstance(desc_resp, dict) and 'output' in desc_resp:
                        desc_embeddings = [item['embedding'] for item in desc_resp['output']['embeddings']]
                    elif hasattr(desc_resp, 'output'):
                        desc_embeddings = [item.embedding for item in desc_resp.output.embeddings]
                    
                    # 处理属性向量
                    if isinstance(attr_resp, dict) and 'output' in attr_resp:
                        attr_embeddings = [item['embedding'] for item in attr_resp['output']['embeddings']]
                    elif hasattr(attr_resp, 'output'):
                        attr_embeddings = [item.embedding for item in attr_resp.output.embeddings]
                    
                    # 存储向量
                    for j, good_id in enumerate(batch_ids):
                        # 融合向量：简单平均
                        fused_embedding = np.mean([
                            np.array(name_embeddings[j]),
                            np.array(desc_embeddings[j]),
                            np.array(attr_embeddings[j])
                        ], axis=0).tolist()
                        
                        self.vector_store[good_id] = {
                            'name_embedding': name_embeddings[j],
                            'desc_embedding': desc_embeddings[j],
                            'attr_embedding': attr_embeddings[j],
                            'fused_embedding': fused_embedding,
                            'name': name_texts[j],
                            'desc': desc_texts[j],
                            'attr': attr_texts[j]
                        }
                    
                    print(f"✅ 完成批量 {i//batch_size + 1}/{total_batches}")
                else:
                    print(f"❌ 批量 {i//batch_size + 1} 生成失败")
            except Exception as e:
                print(f"❌ 批量 {i//batch_size + 1} 处理异常: {e}")
        
        print(f"\n✅ 向量生成完成，共生成 {len(self.vector_store)} 个商品向量")
        
    def save_vector_store(self, save_path="vector_store.pkl"):
        """保存向量存储到文件
        
        Args:
            save_path: 保存文件路径
        """
        if not self.vector_store:
            self.generate_embeddings()
        
        print(f"保存向量存储到 {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(self.vector_store, f)
        print("✅ 向量存储保存完成")
        
    def load_vector_store(self, load_path="vector_store.pkl"):
        """从文件加载向量存储
        
        Args:
            load_path: 加载文件路径
        """
        print(f"从 {load_path} 加载向量存储...")
        with open(load_path, 'rb') as f:
            self.vector_store = pickle.load(f)
        print(f"✅ 向量存储加载完成，共 {len(self.vector_store)} 个商品向量")
        
    def search_similar_goods(self, query_text, top_k=5, vector_type='fused', weights=None):
        """根据查询文本搜索相似商品
        
        Args:
            query_text: 查询文本
            top_k: 返回相似商品数量
            vector_type: 向量类型，可选值：'fused'（融合向量）、'name'（名称向量）、'desc'（描述向量）、'attr'（属性向量）、'hybrid'（混合向量）
            weights: 混合向量权重，格式：{'name': 0.3, 'desc': 0.5, 'attr': 0.2}
            
        Returns:
            相似商品列表
        """
        if not self.vector_store:
            try:
                self.load_vector_store()
            except FileNotFoundError:
                print("⚠️  向量存储文件不存在，开始自动生成...")
                self.generate_embeddings()
                self.save_vector_store()
                print("✅ 向量存储生成完成")
        
        # 生成不同类型的查询向量
        query_vectors = {}
        try:
            # 生成名称向量查询
            name_resp = TextEmbedding.call(
                model=self.embedding_model,
                input=[query_text]
            )
            
            # 生成描述向量查询
            desc_resp = TextEmbedding.call(
                model=self.embedding_model,
                input=[query_text]
            )
            
            # 生成属性向量查询（简化处理）
            attr_resp = TextEmbedding.call(
                model=self.embedding_model,
                input=[query_text]
            )
            
            if (name_resp.status_code == 200 and 
                desc_resp.status_code == 200 and 
                attr_resp.status_code == 200):
                
                # 处理名称向量
                if isinstance(name_resp, dict) and 'output' in name_resp:
                    query_vectors['name'] = name_resp['output']['embeddings'][0]['embedding']
                elif hasattr(name_resp, 'output'):
                    query_vectors['name'] = name_resp.output.embeddings[0].embedding
                
                # 处理描述向量
                if isinstance(desc_resp, dict) and 'output' in desc_resp:
                    query_vectors['desc'] = desc_resp['output']['embeddings'][0]['embedding']
                elif hasattr(desc_resp, 'output'):
                    query_vectors['desc'] = desc_resp.output.embeddings[0].embedding
                
                # 处理属性向量
                if isinstance(attr_resp, dict) and 'output' in attr_resp:
                    query_vectors['attr'] = attr_resp['output']['embeddings'][0]['embedding']
                elif hasattr(attr_resp, 'output'):
                    query_vectors['attr'] = attr_resp.output.embeddings[0].embedding
                
            else:
                raise ValueError(f"查询向量生成失败")
        except Exception as e:
            print(f"❌ 查询向量生成异常: {e}")
            return []
        
        # 计算相似度
        similarities = []
        try:
            for good_id, item in self.vector_store.items():
                similarity = 0.0
                try:
                    if vector_type == 'hybrid':
                        # 混合向量相似度计算
                        if weights is None:
                            # 默认权重
                            weights = {'name': 0.3, 'desc': 0.5, 'attr': 0.2}
                        
                        # 初始化相似度
                        name_sim = 0.0
                        desc_sim = 0.0
                        attr_sim = 0.0
                        
                        # 计算各向量的相似度，添加错误处理
                        try:
                            if 'name_embedding' in item and 'name' in query_vectors:
                                name_sim = np.dot(query_vectors['name'], item['name_embedding']) / (
                                    np.linalg.norm(query_vectors['name']) * np.linalg.norm(item['name_embedding'])
                                )
                        except Exception as e:
                            print(f"⚠️  计算名称向量相似度失败：{e}")
                        
                        try:
                            if 'desc_embedding' in item and 'desc' in query_vectors:
                                desc_sim = np.dot(query_vectors['desc'], item['desc_embedding']) / (
                                    np.linalg.norm(query_vectors['desc']) * np.linalg.norm(item['desc_embedding'])
                                )
                        except Exception as e:
                            print(f"⚠️  计算描述向量相似度失败：{e}")
                        
                        try:
                            if 'attr_embedding' in item and 'attr' in query_vectors:
                                attr_sim = np.dot(query_vectors['attr'], item['attr_embedding']) / (
                                    np.linalg.norm(query_vectors['attr']) * np.linalg.norm(item['attr_embedding'])
                                )
                        except Exception as e:
                            print(f"⚠️  计算属性向量相似度失败：{e}")
                        
                        # 加权融合相似度
                        similarity = (name_sim * weights['name'] + 
                                     desc_sim * weights['desc'] + 
                                     attr_sim * weights['attr'])
                    else:
                        # 单一向量相似度计算
                        vector_key = f'{vector_type}_embedding'
                        # 检查向量类型是否存在，否则尝试其他向量类型
                        if vector_key not in item:
                            # 尝试使用融合向量
                            if 'fused_embedding' in item:
                                vector_key = 'fused_embedding'
                            # 尝试使用名称向量
                            elif 'name_embedding' in item:
                                vector_key = 'name_embedding'
                            # 尝试使用描述向量
                            elif 'desc_embedding' in item:
                                vector_key = 'desc_embedding'
                            # 尝试使用属性向量
                            elif 'attr_embedding' in item:
                                vector_key = 'attr_embedding'
                            # 如果都不存在，跳过当前商品
                            else:
                                continue
                        
                        # 确保查询向量存在
                        if vector_type not in query_vectors:
                            continue
                        
                        similarity = np.dot(query_vectors[vector_type], item[vector_key]) / (
                            np.linalg.norm(query_vectors[vector_type]) * np.linalg.norm(item[vector_key])
                        )
                except Exception as e:
                    print(f"⚠️  计算商品 {good_id} 相似度失败：{e}")
                    continue
                
                similarities.append((good_id, similarity))
        except Exception as e:
            print(f"⚠️  向量相似度计算失败：{e}")
            return []
        
        # 排序并返回top_k个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_goods = similarities[:top_k]
        
        # 获取商品详情
        results = []
        for good_id, similarity in top_goods:
            good_info = self.goods_df[self.goods_df['ID'] == good_id].iloc[0].to_dict()
            results.append({
                '商品ID': good_id,
                '相似度': similarity,
                '商品信息': good_info
            })
        
        return results

if __name__ == "__main__":
    # 示例用法
    vector_store = SemanticVectorStore()
    vector_store.load_data()
    vector_store.generate_embeddings()
    vector_store.save_vector_store()
    
    # 测试相似商品搜索
    print("\n测试相似商品搜索...")
    query = "新鲜水果"
    similar_goods = vector_store.search_similar_goods(query, top_k=3)
    
    print(f"\n查询：{query}")
    print(f"相似商品（前3个）：")
    for i, item in enumerate(similar_goods, 1):
        print(f"{i}. 商品：{item['商品信息']['名称']}，商家：{item['商品信息']['商家']}，相似度：{item['相似度']:.4f}")
