# -*- coding: utf-8 -*-
"""
Fresh Food Recommender
- Multi-modal (text + image) recommendation
- CLIP-based image semantic search
- Knowledge graph integration
- Vector database search
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .semantic_vector_store import SemanticVectorStore
from .multimodal_llm import MultimodalLLM

# Try to import CLIP
CLIP_AVAILABLE = False
try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️  CLIP not available, image search will be limited")

# Try to import knowledge graph
KG_AVAILABLE = False
try:
    from .fresh_knowledge_graph import FreshKnowledgeGraph
    KG_AVAILABLE = True
except ImportError:
    print("⚠️  Knowledge graph not available")

# Get API key from environment
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')

def llm_intent_recognition(user_input: str) -> Dict[str, Any]:
    """LLM-based intent recognition"""
    if not DASHSCOPE_API_KEY:
        return {}
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        prompt = f"""Analyze the user input and extract:
- product_type: specific food item
- tastes: taste preferences
- category: food category
- scene: usage scene

Input: {user_input}
Output JSON:"""
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM intent recognition failed: {e}")
        return {}

def extract_entities_from_image(image_path: str) -> Dict[str, Any]:
    """Extract entities from image using multimodal LLM"""
    try:
        multimodal_llm = MultimodalLLM()
        result = multimodal_llm.extract_product_info(image_path, category="fresh")
        
        # Convert to expected format
        return {
            "product_type": result.get("specific_name", result.get("product_type", "")),
            "tastes": result.get("quality_features", ""),
            "category": result.get("product_type", ""),
            "scene": "daily"
        }
    except Exception as e:
        print(f"Image entity extraction failed: {e}")
        return {}

def call_qwen_generate(prompt: str, conversation_history: List[Dict[str, str]]) -> str:
    """Call Qwen for response generation"""
    if not DASHSCOPE_API_KEY:
        return "API key not available"
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Qwen generation failed: {e}")
        return "AI generation failed"

class FreshFoodRecommender:
    """Fresh food recommendation system"""
    
    def __init__(self):
        self.vector_store = None
        self.kg = None
        self.clip_model = None
        self.clip_processor = None
        self.goods_df = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize components"""
        # Initialize vector store
        try:
            self.vector_store = SemanticVectorStore()
            print("✅ Vector store initialized")
        except Exception as e:
            print(f"⚠️  Vector store initialization failed: {e}")
        
        # Initialize knowledge graph
        if KG_AVAILABLE:
            try:
                self.kg = FreshKnowledgeGraph()
                print("✅ Knowledge graph initialized")
            except Exception as e:
                print(f"⚠️  Knowledge graph initialization failed: {e}")
        
        # Initialize CLIP (使用本地缓存，避免网络下载)
        if CLIP_AVAILABLE:
            try:
                clip_cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sheng", "clip_model_cache")
                local_model_path = os.path.join(clip_cache_dir, "openai-mirror", "clip-vit-base-patch32")
                
                if os.path.exists(local_model_path):
                    # 使用本地模型
                    print(f"使用本地CLIP模型: {local_model_path}")
                    self.clip_model = CLIPModel.from_pretrained(local_model_path)
                    self.clip_processor = CLIPProcessor.from_pretrained(local_model_path)
                    print("✅ 本地CLIP模型初始化成功")
                else:
                    # 本地模型不存在，跳过初始化
                    print(f"⚠️  本地CLIP模型不存在: {local_model_path}")
                    print("⚠️  跳过CLIP初始化，将使用其他推荐方式")
            except Exception as e:
                print(f"⚠️  CLIP初始化失败: {e}")
                print("⚠️  跳过CLIP初始化，将使用其他推荐方式")
        
        # Load goods data
        try:
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "goods_info.csv")
            if os.path.exists(data_path):
                self.goods_df = pd.read_csv(data_path)
                print(f"✅ Loaded {len(self.goods_df)} goods")
            else:
                print("⚠️  Goods data not found")
        except Exception as e:
            print(f"⚠️  Failed to load goods data: {e}")
    
    def _get_kg_client(self):
        """Get knowledge graph client"""
        return self.kg
    
    def _search_by_image(self, image_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search by image using CLIP"""
        if not CLIP_AVAILABLE or not self.clip_model:
            return []
        
        try:
            # Load and process image
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get image features
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features.detach().numpy()
            
            # Get text features for all goods
            if self.goods_df is not None:
                results = []
                for idx, row in self.goods_df.iterrows():
                    text = f"{row.get('名称', '')} {row.get('品类', '')} {row.get('商家', '')}"
                    text_inputs = self.clip_processor(text=[text], return_tensors="pt")
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features.detach().numpy()
                    
                    # Calculate similarity
                    similarity = cosine_similarity(image_features, text_features)[0][0]
                    
                    results.append({
                        "商品主体": row.get('名称', ''),
                        "similar度": float(similarity),
                        "SKU": str(row.get('ID', idx))
                    })
                
                # Sort by similarity
                results.sort(key=lambda x: x.get('similar度', 0), reverse=True)
                return results[:limit]
            
            return []
        except Exception as e:
            print(f"Image search failed: {e}")
            return []
    
    def search_by_text_with_filter(self, query: str, product_type: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search by text with product type filter"""
        try:
            if self.vector_store:
                results = self.vector_store.search(query, top_k=top_k)
                # Filter by product type
                if product_type:
                    results = [r for r in results if product_type in str(r.get('商品主体', ''))]
                return results
            return []
        except Exception as e:
            print(f"Text search with filter failed: {e}")
            return []
    
    def vector_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Vector search"""
        try:
            if self.vector_store:
                return self.vector_store.search(query, top_k=limit)
            return []
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []
    
    def close(self):
        """Close resources"""
        if self.vector_store:
            try:
                self.vector_store.close()
            except:
                pass
        if self.kg:
            try:
                self.kg.close()
            except:
                pass
