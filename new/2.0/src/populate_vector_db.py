#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接填充美妆模块向量数据库
"""

import os
import sys
import pandas as pd
import chromadb
from chromadb.config import Settings

if __name__ == "__main__":
    print("=== 直接填充美妆模块向量数据库 ===")
    print("=" * 60)
    
    try:
        # 设置路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vdb_path = os.path.join(base_dir, "models", "face_vector_db")
        data_path = os.path.join(base_dir, "data", "face_data", "face_goods_info.csv")
        
        print(f"向量数据库路径: {vdb_path}")
        print(f"数据文件路径: {data_path}")
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            sys.exit(1)
        
        # 加载CSV数据
        print(f"从 {data_path} 加载美妆商品数据...")
        data = pd.read_csv(data_path)
        
        print(f"CSV文件列名: {list(data.columns)}")
        print(f"CSV文件行数: {len(data)}")
        
        # 提取数据
        documents = []
        metadatas = []
        ids = []
        
        for _, row in data.iterrows():
            try:
                product_id = str(row['ID'])
                if not product_id:
                    continue
                
                # 构建文档内容
                name = row['名称']
                brand = row['商家']
                category = row['品类']
                price = row['单价']
                
                # 构建文档内容
                document = f"{name} {brand} {category}"
                documents.append(document)
                
                # 构建元数据
                metadatas.append({
                    'product_id': product_id,
                    'name': name,
                    'brand': brand,
                    'category': category,
                    'price': price
                })
                
                # 构建ID
                ids.append(product_id)
            except Exception as e:
                print(f"处理行数据失败: {e}")
                continue
        
        if documents:
            print(f"准备保存 {len(documents)} 个商品到向量数据库")
            
            # 创建向量数据库目录
            os.makedirs(vdb_path, exist_ok=True)
            
            # 初始化chromadb客户端
            client = chromadb.PersistentClient(
                path=vdb_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取或创建集合
            collection = client.get_or_create_collection(name="face_recommendation")
            
            # 清除现有数据（如果有）
            existing_ids = collection.get()['ids']
            if existing_ids:
                collection.delete(ids=existing_ids)
            
            # 添加新数据
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # 验证数据是否添加成功
            doc_count = collection.count()
            print(f"✅ 成功使用chromadb填充向量数据库")
            print(f"✅ 美妆向量库数量已更新为 {doc_count}")
        else:
            print("❌ 没有从CSV文件中提取到有效的商品数据")
            
    except Exception as e:
        print(f"❌ 填充向量数据库失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 填充完成！")
