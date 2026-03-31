#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库与知识图谱融合服务
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from modules.clothing_knowledge_graph import ClothingKnowledgeGraph
from modules.face_knowledge_graph import FaceKnowledgeGraph

# 配置
class Config:
    # 获取项目根目录
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 服装穿搭配置
    CLOTHING_KG_PATH = os.path.join(ROOT_DIR, "models", "clothing_knowledge_graph.pkl")
    CLOTHING_VDB_PATH = os.path.join(ROOT_DIR, "models", "clothing_vector_db")
    CLOTHING_COLLECTION_NAME = "clothing_recommendation"
    
    # 美妆护肤配置
    FACE_KG_PATH = os.path.join(ROOT_DIR, "models", "face_knowledge_graph.pkl")
    FACE_VDB_PATH = os.path.join(ROOT_DIR, "models", "face_vector_db")
    FACE_COLLECTION_NAME = "face_recommendation"
    FACE_DATA_PATH = os.path.join(ROOT_DIR, "data", "face_data", "face_goods_info.csv")
    FACE_VECTOR_DATA_PATH = os.path.join(ROOT_DIR, "data", "face_data", "美妆护肤", "vector_db", "cosmetic_vector_db.json")
    
    DEFAULT_TOP_K = 5
    DEFAULT_STRATEGY = "hybrid"
    WEIGHTS = {"kg": 0.6, "vdb": 0.4}

class FusionService:
    """向量数据库与知识图谱融合服务"""
    
    def __init__(self, category: str = "clothing"):
        """初始化融合服务
        
        Args:
            category: 类别，可选值："clothing"（服装穿搭）、"face"（美妆护肤）
        """
        self.category = category
        
        # 初始化知识图谱
        print("正在加载知识图谱...")
        if category == "clothing":
            self.kg = ClothingKnowledgeGraph.load_graph(Config.CLOTHING_KG_PATH)
        else:
            self.kg = FaceKnowledgeGraph.load_graph(Config.FACE_KG_PATH)
        print("✅ 知识图谱加载完成")
        
        # 初始化向量数据库
        print("正在初始化向量数据库...")
        if category == "clothing":
            vdb_path = Config.CLOTHING_VDB_PATH
            collection_name = Config.CLOTHING_COLLECTION_NAME
        else:
            vdb_path = Config.FACE_VDB_PATH
            collection_name = Config.FACE_COLLECTION_NAME
        
        self.client = chromadb.PersistentClient(
            path=vdb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # 检查向量数据库是否为空，如果为空则使用本地数据
        doc_count = self.collection.count()
        if doc_count == 0:
            print("⚠️  向量数据库为空，正在使用本地数据...")
            # 检查本地向量数据是否存在
            import os
            json_path = os.path.join(vdb_path, "vector_data.json")
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    local_data = json.load(f)
                doc_count = len(local_data)
                print(f"✅ 发现本地向量数据，数量: {doc_count}")
            else:
                print("⚠️  本地向量数据不存在，正在从数据源填充...")
                self._populate_vector_db()
                doc_count = self.collection.count()
        
        print(f"✅ 向量数据库初始化完成，集合文档数量: {doc_count}")
        
        print("🎉 融合服务初始化完成")
    
    def _populate_vector_db(self):
        """从数据源填充向量数据库"""
        if self.category == "face":
            self._populate_face_vector_db()
        else:
            self._populate_clothing_vector_db()
    
    def _populate_face_vector_db(self):
        """填充美妆护肤向量数据库"""
        try:
            # 方法1：尝试从专门的向量数据库JSON文件加载
            if os.path.exists(Config.FACE_VECTOR_DATA_PATH):
                print(f"正在从 {Config.FACE_VECTOR_DATA_PATH} 加载向量数据...")
                with open(Config.FACE_VECTOR_DATA_PATH, 'r', encoding='utf-8') as f:
                    vector_data = json.load(f)
                
                # 提取数据
                ids = []
                documents = []
                metadatas = []
                
                # 检查vector_data的类型
                if isinstance(vector_data, list):
                    for item in vector_data:
                        if isinstance(item, dict):
                            product_id = str(item.get('product_id', item.get('ID', '')))
                            if not product_id:
                                continue
                            
                            # 构建文档内容
                            name = item.get('name', item.get('商品名称', ''))
                            brand = item.get('brand', item.get('商家', ''))
                            category = item.get('category', item.get('品类', ''))
                            ingredients = item.get('ingredients', item.get('成分', ''))
                            efficacy = item.get('efficacy', item.get('功效', ''))
                            
                            document = f"{name} {brand} {category} {ingredients} {efficacy}"
                            
                            ids.append(product_id)
                            documents.append(document)
                            metadatas.append({
                                'product_id': product_id,
                                'name': name,
                                'brand': brand,
                                'category': category,
                                'price': item.get('price', item.get('单价', 0.0))
                            })
                
                if ids:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    print(f"✅ 成功从JSON文件填充 {len(ids)} 个美妆商品到向量数据库")
                    return
            
            # 方法2：尝试从CSV文件加载
            if os.path.exists(Config.FACE_DATA_PATH):
                print(f"正在从 {Config.FACE_DATA_PATH} 加载美妆商品数据...")
                data = pd.read_csv(Config.FACE_DATA_PATH)
                
                ids = []
                documents = []
                metadatas = []
                
                # 打印前几行数据，检查列名
                print(f"CSV文件列名: {list(data.columns)}")
                print(f"CSV文件行数: {len(data)}")
                print(f"前5行数据: {data.head()}")
                
                for _, row in data.iterrows():
                    # 直接使用列索引，确保能够获取到数据
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
                    try:
                        # 分批添加数据，避免一次性添加过多数据
                        batch_size = 100
                        for i in range(0, len(ids), batch_size):
                            batch_ids = ids[i:i+batch_size]
                            batch_documents = documents[i:i+batch_size]
                            batch_metadatas = metadatas[i:i+batch_size]
                            print(f"添加批次 {i//batch_size + 1}/{(len(ids)+batch_size-1)//batch_size}，共 {len(batch_ids)} 个商品")
                            self.collection.add(
                                ids=batch_ids,
                                documents=batch_documents,
                                metadatas=batch_metadatas
                            )
                            print(f"批次 {i//batch_size + 1} 添加成功")
                        print(f"✅ 成功从CSV文件填充 {len(ids)} 个美妆商品到向量数据库")
                        # 验证添加结果
                        doc_count = self.collection.count()
                        print(f"添加后向量数据库文档数量: {doc_count}")
                        return
                    except Exception as e:
                        print(f"❌ 添加数据到向量数据库失败: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ 没有从CSV文件中提取到有效的商品数据")
            
            print("❌ 未找到可用的美妆数据文件")
        except Exception as e:
            print(f"❌ 填充美妆向量数据库失败: {e}")
            # 继续尝试从CSV文件加载
            if os.path.exists(Config.FACE_DATA_PATH):
                print(f"正在从 {Config.FACE_DATA_PATH} 加载美妆商品数据...")
                try:
                    data = pd.read_csv(Config.FACE_DATA_PATH)
                    
                    ids = []
                    documents = []
                    metadatas = []
                    
                    for _, row in data.iterrows():
                        product_id = str(row.get('ID', ''))
                        if not product_id:
                            continue
                        
                        # 构建文档内容
                        name = row.get('名称', '')
                        brand = row.get('商家', '')
                        category = row.get('品类', '')
                        price = row.get('单价', 0.0)
                        
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
                    
                    if ids:
                        self.collection.add(
                            ids=ids,
                            documents=documents,
                            metadatas=metadatas
                        )
                        print(f"✅ 成功从CSV文件填充 {len(ids)} 个美妆商品到向量数据库")
                        return
                except Exception as e2:
                    print(f"❌ 从CSV文件加载失败: {e2}")
    
    def _populate_clothing_vector_db(self):
        """填充服装穿搭向量数据库"""
        # 这里可以添加服装数据的填充逻辑
        print("服装向量数据库填充功能待实现")
    
    def query(self, query_text: str, top_k: int = Config.DEFAULT_TOP_K, 
              strategy: str = Config.DEFAULT_STRATEGY) -> List[Dict]:
        """统一查询接口"""
        if strategy == "kg_only":
            return self._kg_query(query_text, top_k)
        elif strategy == "vdb_only":
            return self._vdb_query(query_text, top_k)
        else:
            return self._hybrid_query(query_text, top_k)
    
    def _kg_query(self, query_text: str, top_k: int) -> List[Dict]:
        """知识图谱查询"""
        linked_entities = self.kg.entity_linking(query_text)
        
        results = []
        
        if self.category == "clothing":
            # 服装穿搭查询
            for entity in linked_entities:
                if entity["实体类型"] == "品类":
                    goods = self.kg.query_goods_by_category(entity["实体名称"])
                    results.extend(goods)
                elif entity["实体类型"] == "风格":
                    goods = self.kg.query_goods_by_style(entity["实体名称"])
                    results.extend(goods)
                elif entity["实体类型"] == "季节":
                    goods = self.kg.query_goods_by_season(entity["实体名称"])
                    results.extend(goods)
            
            # 去重
            seen_ids = set()
            unique_results = []
            for item in results:
                if item.get("商品ID") not in seen_ids:
                    seen_ids.add(item.get("商品ID"))
                    unique_results.append({
                        "product_id": item.get("商品ID"),
                        "name": item.get("商品名称"),
                        "price": item.get("售价", 0.0),
                        "score": 1.0,
                        "source": "knowledge_graph"
                    })
        else:
            # 美妆护肤查询
            # 使用get_user_recommendations方法获取推荐
            recommendations = self.kg.get_user_recommendations('user_1', top_k, query_text)
            unique_results = []
            seen_ids = set()
            for rec in recommendations:
                if rec.get('product_id') not in seen_ids:
                    seen_ids.add(rec.get('product_id'))
                    unique_results.append({
                        "product_id": rec.get('product_id'),
                        "name": rec.get('name', ''),
                        "price": rec.get('price', 0.0),
                        "score": rec.get('score', 0.0),
                        "source": "knowledge_graph"
                    })
        
        return unique_results[:top_k]
    
    def _vdb_query(self, query_text: str, top_k: int) -> List[Dict]:
        """向量数据库查询"""
        try:
            # 尝试使用chromadb的查询功能
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted = []
            for i, id in enumerate(results["ids"][0]):
                formatted.append({
                    "product_id": id,
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "source": "vector_db"
                })
            
            return formatted[:top_k]
        except Exception as e:
            print(f"❌ 向量数据库查询失败，使用本地数据: {e}")
            # 如果查询失败，使用本地保存的向量数据
            return self._local_vdb_query(query_text, top_k)
    
    def _local_vdb_query(self, query_text: str, top_k: int) -> List[Dict]:
        """使用本地向量数据进行查询"""
        import json
        import os
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 加载本地向量数据
        vdb_path = Config.FACE_VDB_PATH if self.category == "face" else Config.CLOTHING_VDB_PATH
        json_path = os.path.join(vdb_path, "vector_data.json")
        
        if not os.path.exists(json_path):
            print(f"❌ 本地向量数据文件不存在: {json_path}")
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            vector_data = json.load(f)
        
        # 提取文档
        documents = [item['document'] for item in vector_data]
        product_ids = [item['product_id'] for item in vector_data]
        
        # 使用TF-IDF进行文本相似度计算
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # 计算查询文本的TF-IDF向量
        query_vector = vectorizer.transform([query_text])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # 排序并获取top_k结果
        sorted_indices = similarities.argsort()[::-1][:top_k]
        
        formatted = []
        for idx in sorted_indices:
            item = vector_data[idx]
            formatted.append({
                "product_id": item['product_id'],
                "text": item['document'],
                "score": float(similarities[idx]),
                "source": "local_vector_db"
            })
        
        return formatted
    
    def _hybrid_query(self, query_text: str, top_k: int) -> List[Dict]:
        """混合查询"""
        # 并行查询
        kg_results = self._kg_query(query_text, top_k)
        vdb_results = self._vdb_query(query_text, top_k)
        
        # 融合结果
        return self._merge_results(kg_results, vdb_results)
    
    def _merge_results(self, kg_results: List[Dict], vdb_results: List[Dict]) -> List[Dict]:
        """融合结果"""
        # 统一数据格式
        kg_formatted = []
        for item in kg_results:
            kg_formatted.append({
                "product_id": item["product_id"],
                "name": item.get("name", ""),
                "text": item.get("name", ""),
                "price": item.get("price", 0.0),
                "score": item["score"] * Config.WEIGHTS["kg"],
                "source": "knowledge_graph"
            })
        
        vdb_formatted = []
        for item in vdb_results:
            vdb_formatted.append({
                "product_id": item["product_id"],
                "name": "",
                "text": item["text"],
                "price": 0.0,
                "score": item["score"] * Config.WEIGHTS["vdb"],
                "source": "vector_db"
            })
        
        # 去重和融合
        merged = []
        seen_ids = set()
        
        # 添加知识图谱结果
        for result in kg_formatted:
            if result["product_id"] not in seen_ids:
                seen_ids.add(result["product_id"])
                merged.append(result)
        
        # 添加向量数据库结果
        for result in vdb_formatted:
            if result["product_id"] not in seen_ids:
                seen_ids.add(result["product_id"])
                merged.append(result)
            else:
                # 更新已有结果的分数
                for item in merged:
                    if item["product_id"] == result["product_id"]:
                        item["score"] += result["score"]
                        item["text"] += " " + result["text"]
                        break
        
        # 排序
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged[:len(kg_results) + len(vdb_results)]
    
    def entity_linking(self, text: str) -> List[Dict]:
        """实体链接"""
        return self.kg.entity_linking(text)

# 测试代码
if __name__ == "__main__":
    print("=== 融合服务测试 - 服装穿搭 ===")
    clothing_service = FusionService(category="clothing")
    
    # 测试服装穿搭查询
    print("\n1. 混合查询（知识图谱+向量数据库）：")
    results = clothing_service.query("适合夏天的T恤", top_k=3)
    for i, result in enumerate(results[:3]):
        print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}, 来源: {result['source']}")
    
    print("\n2. 知识图谱查询：")
    results = clothing_service.query("适合夏天的T恤", top_k=2, strategy="kg_only")
    for i, result in enumerate(results[:2]):
        print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}")
    
    print("\n3. 向量数据库查询：")
    results = clothing_service.query("适合夏天的T恤", top_k=2, strategy="vdb_only")
    for i, result in enumerate(results[:2]):
        print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}")
    
    print("\n" + "=" * 60)
    print("=== 融合服务测试 - 美妆护肤 ===")
    print("=" * 60)
    
    try:
        face_service = FusionService(category="face")
        
        # 测试美妆护肤查询
        print("\n1. 混合查询（知识图谱+向量数据库）：")
        results = face_service.query("补水保湿的面霜", top_k=3)
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}, 来源: {result['source']}")
        
        print("\n2. 知识图谱查询：")
        results = face_service.query("补水保湿的面霜", top_k=2, strategy="kg_only")
        for i, result in enumerate(results[:2]):
            print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}")
        
        print("\n3. 向量数据库查询：")
        results = face_service.query("补水保湿的面霜", top_k=2, strategy="vdb_only")
        for i, result in enumerate(results[:2]):
            print(f"   {i+1}. ID: {result['product_id']}, 分数: {result['score']:.4f}")
    except Exception as e:
        print(f"❌ 美妆护肤模块测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 融合服务测试完成！")