#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品数据库管理模块
用于管理商品数据的存储、查询和相似物品推荐
"""

import sqlite3
import pandas as pd
import os
import json
from typing import List, Dict, Any

# 数据库文件路径
GOODS_DB_PATH = "ecommerce_goods.db"

class GoodsDatabase:
    """商品数据库管理类"""
    
    def __init__(self):
        """初始化商品数据库"""
        self.conn = None
        self.cursor = None
        self._init_db()
    
    def _init_db(self):
        """初始化商品数据库表结构"""
        try:
            self.conn = sqlite3.connect(GOODS_DB_PATH)
            self.cursor = self.conn.cursor()
            
            # 创建商品表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS goods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goods_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                brand TEXT,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                description TEXT,
                rating REAL DEFAULT 0,
                review_count INTEGER DEFAULT 0,
                sales_count INTEGER DEFAULT 0,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 创建商品特征表（用于相似物品推荐）
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS goods_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goods_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value TEXT NOT NULL,
                FOREIGN KEY (goods_id) REFERENCES goods(goods_id)
            )
            ''')
            
            # 创建商品索引
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_goods_category ON goods(category)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_goods_brand ON goods(brand)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_goods_features_goods_id ON goods_features(goods_id)')
            
            self.conn.commit()
            print("✅ 商品数据库初始化成功")
            
        except Exception as e:
            print(f"❌ 初始化商品数据库失败: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
            self.cursor = None
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def import_goods_from_csv(self, csv_path: str, category: str = None):
        """从CSV文件导入商品数据
        
        Args:
            csv_path: CSV文件路径
            category: 商品类别（可选）
        
        Returns:
            int: 导入的商品数量
        """
        if not self.conn:
            print("❌ 数据库连接未初始化")
            return 0
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            print(f"📊 读取CSV文件，共 {len(df)} 条记录")
            
            imported_count = 0
            
            for idx, row in df.iterrows():
                try:
                    # 构建商品数据
                    goods_data = {
                        'goods_id': str(row.get('ID', row.get('商品ID', f"{category}_{idx}")),
                        'name': row.get('名称', row.get('商品名称', '')),
                        'brand': row.get('商家', row.get('品牌', '')),
                        'category': row.get('品类', row.get('类别', category or '其他')),
                        'price': float(row.get('单价', row.get('价格', 0))),
                        'description': row.get('评价', row.get('描述', '')),
                        'rating': float(row.get('评分', 0)),
                        'review_count': int(row.get('评价数', 0)),
                        'sales_count': int(row.get('销量', 0))
                    }
                    
                    # 插入商品数据
                    self._insert_goods(goods_data)
                    imported_count += 1
                    
                    # 构建商品特征
                    features = {
                        'category': goods_data['category'],
                        'brand': goods_data['brand'],
                        'price_range': self._get_price_range(goods_data['price'])
                    }
                    
                    # 插入商品特征
                    for feature_name, feature_value in features.items():
                        self._insert_goods_feature(goods_data['goods_id'], feature_name, feature_value)
                    
                except Exception as e:
                    print(f"❌ 导入第 {idx} 条商品失败: {e}")
                    continue
            
            self.conn.commit()
            print(f"✅ 成功导入 {imported_count} 条商品数据")
            return imported_count
            
        except Exception as e:
            print(f"❌ 从CSV导入商品失败: {e}")
            return 0
    
    def _insert_goods(self, goods_data: Dict[str, Any]):
        """插入商品数据"""
        try:
            # 检查商品是否已存在
            self.cursor.execute("SELECT id FROM goods WHERE goods_id = ?", (goods_data['goods_id'],))
            if self.cursor.fetchone():
                # 更新现有商品
                self.cursor.execute('''
                UPDATE goods SET
                    name = ?,
                    brand = ?,
                    category = ?,
                    price = ?,
                    description = ?,
                    rating = ?,
                    review_count = ?,
                    sales_count = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE goods_id = ?
                ''', (
                    goods_data['name'],
                    goods_data['brand'],
                    goods_data['category'],
                    goods_data['price'],
                    goods_data['description'],
                    goods_data['rating'],
                    goods_data['review_count'],
                    goods_data['sales_count'],
                    goods_data['goods_id']
                ))
            else:
                # 插入新商品
                self.cursor.execute('''
                INSERT INTO goods (
                    goods_id, name, brand, category, price, description, 
                    rating, review_count, sales_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    goods_data['goods_id'],
                    goods_data['name'],
                    goods_data['brand'],
                    goods_data['category'],
                    goods_data['price'],
                    goods_data['description'],
                    goods_data['rating'],
                    goods_data['review_count'],
                    goods_data['sales_count']
                ))
        except Exception as e:
            print(f"❌ 插入商品数据失败: {e}")
            raise
    
    def _insert_goods_feature(self, goods_id: str, feature_name: str, feature_value: str):
        """插入商品特征"""
        try:
            # 检查特征是否已存在
            self.cursor.execute('''
            SELECT id FROM goods_features 
            WHERE goods_id = ? AND feature_name = ? AND feature_value = ?
            ''', (goods_id, feature_name, feature_value))
            
            if not self.cursor.fetchone():
                self.cursor.execute('''
                INSERT INTO goods_features (goods_id, feature_name, feature_value)
                VALUES (?, ?, ?)
                ''', (goods_id, feature_name, feature_value))
        except Exception as e:
            print(f"❌ 插入商品特征失败: {e}")
    
    def _get_price_range(self, price: float) -> str:
        """获取价格区间"""
        if price < 50:
            return '低价'
        elif price < 200:
            return '中低价'
        elif price < 500:
            return '中价'
        elif price < 1000:
            return '中高价'
        else:
            return '高价'
    
    def get_goods_by_category(self, category: str, limit: int = 100):
        """根据类别获取商品
        
        Args:
            category: 商品类别
            limit: 返回数量限制
        
        Returns:
            List[Dict[str, Any]]: 商品列表
        """
        if not self.conn:
            return []
        
        try:
            self.cursor.execute('''
            SELECT goods_id, name, brand, category, price, description, rating, review_count, sales_count
            FROM goods 
            WHERE category = ? 
            ORDER BY sales_count DESC, rating DESC 
            LIMIT ?
            ''', (category, limit))
            
            goods_list = []
            for row in self.cursor.fetchall():
                goods_list.append({
                    'goods_id': row[0],
                    'name': row[1],
                    'brand': row[2],
                    'category': row[3],
                    'price': row[4],
                    'description': row[5],
                    'rating': row[6],
                    'review_count': row[7],
                    'sales_count': row[8]
                })
            
            return goods_list
            
        except Exception as e:
            print(f"❌ 根据类别获取商品失败: {e}")
            return []
    
    def search_goods(self, query: str, category: str = None, limit: int = 50):
        """搜索商品
        
        Args:
            query: 搜索关键词
            category: 商品类别（可选）
            limit: 返回数量限制
        
        Returns:
            List[Dict[str, Any]]: 商品列表
        """
        if not self.conn:
            return []
        
        try:
            if category:
                self.cursor.execute('''
                SELECT goods_id, name, brand, category, price, description, rating, review_count, sales_count
                FROM goods 
                WHERE (name LIKE ? OR description LIKE ?) AND category = ?
                ORDER BY sales_count DESC, rating DESC 
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', category, limit))
            else:
                self.cursor.execute('''
                SELECT goods_id, name, brand, category, price, description, rating, review_count, sales_count
                FROM goods 
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY sales_count DESC, rating DESC 
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
            
            goods_list = []
            for row in self.cursor.fetchall():
                goods_list.append({
                    'goods_id': row[0],
                    'name': row[1],
                    'brand': row[2],
                    'category': row[3],
                    'price': row[4],
                    'description': row[5],
                    'rating': row[6],
                    'review_count': row[7],
                    'sales_count': row[8]
                })
            
            return goods_list
            
        except Exception as e:
            print(f"❌ 搜索商品失败: {e}")
            return []
    
    def get_similar_goods(self, goods_id: str, limit: int = 10):
        """获取相似商品
        
        Args:
            goods_id: 商品ID
            limit: 返回数量限制
        
        Returns:
            List[Dict[str, Any]]: 相似商品列表
        """
        if not self.conn:
            return []
        
        try:
            # 获取目标商品信息
            self.cursor.execute('''
            SELECT goods_id, name, brand, category, price
            FROM goods 
            WHERE goods_id = ?
            ''', (goods_id,))
            target_goods = self.cursor.fetchone()
            
            if not target_goods:
                print(f"❌ 商品 {goods_id} 不存在")
                return []
            
            target_brand = target_goods[2]
            target_category = target_goods[3]
            target_price = target_goods[4]
            price_range = self._get_price_range(target_price)
            
            # 搜索相似商品（基于品牌、类别和价格区间）
            self.cursor.execute('''
            SELECT g.goods_id, g.name, g.brand, g.category, g.price, g.description, g.rating, g.review_count, g.sales_count
            FROM goods g
            JOIN goods_features gf ON g.goods_id = gf.goods_id
            WHERE g.goods_id != ?
            AND (
                g.brand = ? OR 
                g.category = ? OR 
                gf.feature_name = 'price_range' AND gf.feature_value = ?
            )
            GROUP BY g.goods_id
            ORDER BY 
                CASE 
                    WHEN g.brand = ? THEN 3
                    WHEN g.category = ? THEN 2
                    WHEN gf.feature_name = 'price_range' AND gf.feature_value = ? THEN 1
                    ELSE 0
                END DESC,
                g.sales_count DESC, 
                g.rating DESC
            LIMIT ?
            ''', (goods_id, target_brand, target_category, price_range, target_brand, target_category, price_range, limit))
            
            similar_goods = []
            for row in self.cursor.fetchall():
                similar_goods.append({
                    'goods_id': row[0],
                    'name': row[1],
                    'brand': row[2],
                    'category': row[3],
                    'price': row[4],
                    'description': row[5],
                    'rating': row[6],
                    'review_count': row[7],
                    'sales_count': row[8]
                })
            
            return similar_goods
            
        except Exception as e:
            print(f"❌ 获取相似商品失败: {e}")
            return []
    
    def get_goods_by_id(self, goods_id: str):
        """根据ID获取商品
        
        Args:
            goods_id: 商品ID
        
        Returns:
            Dict[str, Any]: 商品信息
        """
        if not self.conn:
            return None
        
        try:
            self.cursor.execute('''
            SELECT goods_id, name, brand, category, price, description, rating, review_count, sales_count
            FROM goods 
            WHERE goods_id = ?
            ''', (goods_id,))
            
            row = self.cursor.fetchone()
            if row:
                return {
                    'goods_id': row[0],
                    'name': row[1],
                    'brand': row[2],
                    'category': row[3],
                    'price': row[4],
                    'description': row[5],
                    'rating': row[6],
                    'review_count': row[7],
                    'sales_count': row[8]
                }
            return None
            
        except Exception as e:
            print(f"❌ 根据ID获取商品失败: {e}")
            return None
    
    def get_recommendations(self, user_needs: str, category: str, limit: int = 10):
        """获取推荐商品
        
        Args:
            user_needs: 用户需求
            category: 商品类别
            limit: 返回数量限制
        
        Returns:
            List[Dict[str, Any]]: 推荐商品列表
        """
        if not self.conn:
            return []
        
        try:
            # 首先尝试根据用户需求搜索
            search_results = self.search_goods(user_needs, category, limit)
            
            # 如果搜索结果不足，补充热门商品
            if len(search_results) < limit:
                hot_goods = self.get_goods_by_category(category, limit - len(search_results))
                # 去重
                existing_ids = {item['goods_id'] for item in search_results}
                for goods in hot_goods:
                    if goods['goods_id'] not in existing_ids:
                        search_results.append(goods)
                        existing_ids.add(goods['goods_id'])
                        if len(search_results) >= limit:
                            break
            
            return search_results[:limit]
            
        except Exception as e:
            print(f"❌ 获取推荐商品失败: {e}")
            # 失败时返回热门商品
            return self.get_goods_by_category(category, limit)

# 全局商品数据库实例
goods_db = None

def get_goods_db():
    """获取商品数据库实例"""
    global goods_db
    if goods_db is None:
        goods_db = GoodsDatabase()
    return goods_db

# 初始化商品数据库
if not os.path.exists(GOODS_DB_PATH):
    goods_db = GoodsDatabase()
    goods_db.close()
