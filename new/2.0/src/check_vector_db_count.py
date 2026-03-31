#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查美妆模块向量数据库文档数量
"""

import os
import sys
import chromadb
from chromadb.config import Settings

if __name__ == "__main__":
    # 设置向量数据库路径
    vdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_vector_db")
    collection_name = "face_recommendation"
    
    print(f"检查向量数据库路径: {vdb_path}")
    
    # 初始化向量数据库客户端
    client = chromadb.PersistentClient(
        path=vdb_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 获取或创建集合
    collection = client.get_or_create_collection(name=collection_name)
    
    # 获取文档数量
    doc_count = collection.count()
    print(f"向量数据库文档数量: {doc_count}")
    
    if doc_count > 0:
        print("✅ 美妆向量库数量不为0，修复成功！")
    else:
        print("⚠️  美妆向量库数量仍为0，需要进一步检查")
    
    print("✅ 检查完成！")
