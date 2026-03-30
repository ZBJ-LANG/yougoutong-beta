#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态embedding模块
使用CLIP模型进行图像-文本编码
"""

import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Union
import chromadb
from chromadb.config import Settings

class MultimodalEmbedder:
    """多模态embedding编码器"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称，可选 ViT-B/32, ViT-B/16, ViT-L/14
        """
        print(f"正在加载CLIP模型: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"✅ CLIP模型加载完成，使用设备: {self.device}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片embedding向量
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """批量编码图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            图片embedding矩阵
        """
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(self.preprocess(image))
        
        image_input = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本
        
        Args:
            text: 文本内容
            
        Returns:
            文本embedding向量
        """
        text_input = clip.tokenize(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        
        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本embedding矩阵
        """
        text_inputs = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        
        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray, 
                          target_embeddings: np.ndarray) -> np.ndarray:
        """计算余弦相似度
        
        Args:
            query_embedding: 查询向量
            target_embeddings: 目标向量矩阵
            
        Returns:
            相似度数组
        """
        return np.dot(target_embeddings, query_embedding)


class MultimodalVectorDB:
    """多模态向量数据库"""
    
    def __init__(self, 
                 collection_name: str = "multimodal_products",
                 persist_directory: str = "./multimodal_vector_db"):
        """初始化多模态向量数据库
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.embedder = MultimodalEmbedder()
        
        # 初始化Chroma DB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ 多模态向量数据库初始化完成，集合: {collection_name}")
    
    def add_product(self, 
                    product_id: str, 
                    image_path: str, 
                    text: str,
                    metadata: Dict[str, Any] = None):
        """添加商品到向量数据库
        
        Args:
            product_id: 商品ID
            image_path: 商品图片路径
            text: 商品文本描述
            metadata: 其他元数据
        """
        # 编码图片和文本
        image_embedding = self.embedder.encode_image(image_path)
        text_embedding = self.embedder.encode_text(text)
        
        # 融合embedding (简单平均)
        combined_embedding = (image_embedding + text_embedding) / 2
        
        # 元数据
        meta = metadata or {}
        meta["text"] = text
        
        # 存储到向量数据库
        self.collection.add(
            ids=[product_id],
            embeddings=[combined_embedding.tolist()],
            documents=[text],
            metadatas=[meta]
        )
        
        print(f"✅ 已添加商品: {product_id}")
    
    def search(self, 
               query: str = None,
               query_image_path: str = None,
               top_k: int = 5) -> List[Dict]:
        """多模态搜索
        
        Args:
            query: 文本查询
            query_image_path: 图片查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 生成查询embedding
        query_embeddings = []
        
        if query:
            text_embedding = self.embedder.encode_text(query)
            query_embeddings.append(text_embedding)
        
        if query_image_path:
            image_embedding = self.embedder.encode_image(query_image_path)
            query_embeddings.append(image_embedding)
        
        if not query_embeddings:
            raise ValueError("必须提供文本查询或图片查询")
        
        # 融合查询embedding
        query_embedding = np.mean(query_embeddings, axis=0)
        
        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "product_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]
            })
        
        return formatted_results
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """以图搜图
        
        Args:
            image_path: 查询图片路径
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        return self.search(query_image_path=image_path, top_k=top_k)
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """文本搜索
        
        Args:
            query: 文本查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        return self.search(query=query, top_k=top_k)


# 测试代码
if __name__ == "__main__":
    # 初始化多模态向量数据库
    vdb = MultimodalVectorDB(
        collection_name="test_products",
        persist_directory="./test_multimodal_db"
    )
    
    # 添加测试商品
    test_products = [
        {
            "id": "product_001",
            "image": "test_images/t-shirt.jpg",
            "text": "纯棉T恤 夏季透气 简约时尚",
            "metadata": {"category": "clothing", "price": 99.0}
        },
        {
            "id": "product_002", 
            "image": "test_images/dress.jpg",
            "text": "碎花连衣裙 春夏新款 甜美风格",
            "metadata": {"category": "clothing", "price": 199.0}
        }
    ]
    
    for product in test_products:
        if os.path.exists(product["image"]):
            vdb.add_product(
                product_id=product["id"],
                image_path=product["image"],
                text=product["text"],
                metadata=product["metadata"]
            )
    
    # 文本搜索
    print("\n=== 文本搜索: '夏季T恤' ===")
    results = vdb.search_by_text("夏季T恤")
    for r in results:
        print(f"- {r['product_id']}: {r['text']} (相似度: {r['similarity']:.4f})")
    
    # 图片搜索
    print("\n=== 图片搜索 ===")
    if os.path.exists("test_images/query.jpg"):
        results = vdb.search_by_image("test_images/query.jpg")
        for r in results:
            print(f"- {r['product_id']}: {r['text']} (相似度: {r['similarity']:.4f})")
