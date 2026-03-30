# -*- coding: utf-8 -*-
"""
Fresh Food Recommendation Agent
- Tool selection based on input type
- Multi-step reasoning
- Result quality reflection
- Conversation memory
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from enum import Enum
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

from .fresh_food_recommender import (
    llm_intent_recognition,
    extract_entities_from_image,
    call_qwen_generate,
    FreshFoodRecommender,
    DASHSCOPE_API_KEY
)
from openai import OpenAI


class FreshFoodAgent:
    
    def __init__(self):
        self.conversation_history = []
        self.recommender = FreshFoodRecommender()
        
    def _rule_intent(self, user_input):
        s = user_input
        
        product_keywords = [
            '苹果', '香蕉', '荔枝', '樱桃', '草莓', '葡萄', '蓝莓',
            '猕猴桃', '橙子', '豆腐', '鸡肉', '牛肉', '猪肉', '羊肉',
            '鱼', '三文鱼', '鸡蛋', '鸭蛋', '黄瓜', '西瓜', '西红柿',
            '青椒', '菠萝', '梨', '哈密瓜', '虾', '蟹', '豆浆', '豆干',
            '柚子', '桃子', '火龙果', '鲈鱼', '鳜鱼', '生蚝', '花蛤',
            '土豆', '山药', '茄子', '白菜', '生菜', '菠菜', '西兰花',
            '胡萝卜', '莲藕', '南瓜', '腐竹', '豆皮', '鹅蛋', '鹌鹑蛋',
        ]
        
        taste_keywords = [
            '脆', '甜', '酸', '软', '鲜', '嫩', '多汁',
            '清甜', '脆甜', '酸甜', '香甜', '鲜甜',
            '清香', '鲜嫩', '滑嫩', '细嫩', '细腻',
            '绵密', '绵软', '软糯', '沙糯', '爽口', '清脆', '香嫩',
            '筋道', '粉面', '肥美', '醇香', 'Q弹',
        ]
        
        products = []
        for kw in product_keywords:
            if kw in s:
                products.append(kw)
                
        tastes = []
        for kw in taste_keywords:
            if kw in s:
                tastes.append(kw)
        
        return {
            "product_type": products[0] if products else "",
            "tastes": tastes,
            "category": "",
            "scene": "daily"
        }
    
    def think(self, user_input, image_path=None):
        print("\n" + "=" * 60)
        print("  Agent Thinking Process")
        print("=" * 60)
        
        thought_log = []
        
        print("\n[Step 1] Understanding Input...")
        entities = {}
        vector_results = []
        
        if image_path and os.path.exists(image_path):
            print("  - Input type: Image")
            print("  - Using CLIP for image semantic search...")
            vector_results = self.recommender._search_by_image(image_path, limit=20)
            
            print("  - Using Qwen VL for entity extraction (multi-modal fusion)...")
            img_entities = extract_entities_from_image(image_path)
            entities = {
                "product_type": img_entities.get("product_type", ""),
                "tastes": img_entities.get("tastes", []),
                "category": img_entities.get("category", ""),
                "scene": img_entities.get("scene", ""),
                "input_type": "Image"
            }
            
            product_type = entities.get("product_type", "")
            if product_type and vector_results:
                clip_types = [r.get('商品主体','') for r in vector_results]
                print(f"  - CLIP found: {clip_types[:5]}")
                print(f"  - Qwen VL detected: {product_type}")
                
                filtered = [r for r in vector_results if r.get('商品主体') == product_type]
                if filtered:
                    print(f"  - Fused result: {len(filtered)} products (CLIP + VL)")
                    vector_results = filtered
        else:
            print("  - Input type: Text")
            print("  - Using rule-based intent recognition...")
            entities = self._rule_intent(user_input)
            entities["input_type"] = "Text"
            
            print("  - Trying LLM intent recognition as backup...")
            try:
                llm_result = llm_intent_recognition(user_input)
                if llm_result and llm_result.get("product_type"):
                    entities = {
                        "product_type": llm_result.get("product_type", ""),
                        "tastes": llm_result.get("tastes", []),
                        "category": llm_result.get("category", ""),
                        "scene": llm_result.get("scene", ""),
                        "input_type": "Text"
                    }
            except Exception as e:
                print(f"    LLM failed: {e}")
        
        thought_log.append({"step": "understanding", "entities": entities})
        print(f"\n  Extracted: type={entities.get('product_type')}, tastes={entities.get('tastes')}")
        
        print("\n[Step 2] Reasoning and Planning...")
        product_type = entities.get("product_type", "")
        tastes = entities.get("tastes", [])
        
        if not product_type:
            print("  - No product type found, using vector search only")
            product_type = ""
        
        print(f"  - Plan: search for {product_type} with tastes {tastes}")
        thought_log.append({"step": "reasoning", "product_type": product_type, "tastes": tastes})
        
        print("\n[Step 3] Executing Search...")
        kg_results = []
        kg = self.recommender._get_kg_client()
        if kg:
            try:
                if tastes:
                    for t in tastes[:2]:
                        r = kg.search_by_product_type_and_taste(product_type, t, limit=10)
                        kg_results.extend(r)
                else:
                    kg_results = kg.search_by_product_type(product_type, limit=10)
                print(f"  - KG found {len(kg_results)} results")
            except Exception as e:
                print(f"  - KG failed: {e}")
        
        if not vector_results:
            try:
                use_filter = bool(product_type)
                if use_filter:
                    vector_results = self.recommender.search_by_text_with_filter(user_input, product_type=product_type, top_k=20)
                else:
                    vector_results = self.recommender.vector_search(user_input, limit=20)
                print(f"  - Vector found {len(vector_results)} results")
            except Exception as e:
                print(f"  - Vector failed: {e}")
        
        thought_log.append({"step": "executing", "kg": len(kg_results), "vec": len(vector_results)})
        
        if not kg_results and vector_results and product_type:
            filtered = [r for r in vector_results if r.get('商品主体') == product_type or product_type in str(r.get('商品主体', ''))]
            if filtered:
                print(f"  - Filtered {len(filtered)} results by product_type={product_type}")
                vector_results = filtered
            else:
                print(f"  - No product_type filter match, {len(vector_results)} vector results used as-is")
        
        print("\n[Step 4] Merging Results...")
        all_products = {}
        for r in kg_results:
            sku = r.get("SKU") or r.get("sku") or ""
            if sku and sku not in all_products:
                all_products[sku] = {"source": "KG", **r, "score": 1.0}
        
        for r in vector_results:
            sku = r.get("SKU") or r.get("sku") or ""
            if sku:
                if sku in all_products:
                    all_products[sku]["source"] = "KG+Vec"
                    all_products[sku]["score"] = 1.0 + (all_products[sku].get("similar度") or 0)
                else:
                    all_products[sku] = {"source": "Vec", **r}
        
        products = sorted(all_products.values(), key=lambda x: x.get("score", 0), reverse=True)[:10]
        print(f"  - Merged to {len(products)} products")
        thought_log.append({"step": "merging", "total": len(products)})
        
        print("\n[Step 5] Generating Response...")
        if products:
            def clean(obj):
                if isinstance(obj, str):
                    return ''.join(c for c in obj if not (0xD800 <= ord(c) <= 0xDFFF) and ord(c) >= 0x20)
                elif isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean(i) for i in obj]
                return obj
            
            prompt = f"""User: {user_input}\n\nResults:\n{json.dumps(clean(products[:10]), ensure_ascii=False, indent=2)}\n\nRecommend products."""
            response = call_qwen_generate(prompt, self.conversation_history)
        else:
            response = "No products found."
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "thought_log": thought_log,
            "entities": entities,
            "search_results": products,
            "response": response
        }
    
    def run(self, user_input, image_path=None):
        result = self.think(user_input, image_path)
        
        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        for log in result["thought_log"]:
            print(f"\n[{log.get('step', '').upper()}]")
            for k, v in log.items():
                if k != 'step':
                    print(f"  {k}: {v}")
        
        return result["response"]
    
    def close(self):
        if self.recommender:
            self.recommender.close()


def choose_image_file():
    """交互式选择图片文件"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="选择生鲜商品图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("所有文件", "*.*")
        ]
    )
    root.destroy()
    return file_path if file_path else None


def main():
    agent = FreshFoodAgent()
    current_image = None
    
    print("\n" + "=" * 50)
    print("  生鲜电商多模态推荐系统")
    print("  输入文字或上传图片进行推荐")
    print("=" * 50)
    print("\n命令说明:")
    print("  /upload  - 上传图片文件 (图形对话框)")
    print("  /file    - 输入图片路径")
    print("  /clear   - 清除当前图片")
    print("  /status  - 查看当前状态")
    print("  直接输入文字 - 进行文本推荐")
    print("  quit     - 退出程序")
    print("=" * 50)
    
    while True:
        try:
            prompt = f"[User{(' - ' + os.path.basename(current_image)) if current_image else ''}] "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', 'exit', '/quit']:
                break
            
            if user_input == '/upload':
                print("\n[系统] 正在打开文件选择器...")
                img_path = choose_image_file()
                if img_path:
                    current_image = img_path
                    print(f"[系统] 已选择图片: {img_path}")
                else:
                    print("[系统] 未选择文件")
                continue
            
            elif user_input == '/file':
                img_path = input("请输入图片路径: ").strip().strip('"').strip("'")
                if os.path.exists(img_path):
                    current_image = img_path
                    print(f"[系统] 已设置图片: {img_path}")
                else:
                    print(f"[系统] 文件不存在: {img_path}")
                continue
            
            elif user_input == '/clear':
                current_image = None
                print("[系统] 已清除图片")
                continue
            
            elif user_input == '/status':
                print(f"\n[状态]")
                print(f"  当前图片: {current_image if current_image else '无'}")
                print(f"  输入模式: {'图片' if current_image else '文本'}")
                continue
            
            elif user_input.startswith('/image '):
                img = user_input[7:].strip()
                if os.path.exists(img):
                    current_image = img
                    print(f"[系统] 已设置图片: {img}")
                else:
                    print(f"[系统] 文件不存在: {img}")
                continue
            
            else:
                if current_image:
                    resp = agent.run(user_input, image_path=current_image)
                else:
                    resp = agent.run(user_input)
            
            print(f"\n[Assistant]\n{resp}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
    
    agent.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
