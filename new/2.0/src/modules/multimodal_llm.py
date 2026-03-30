#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态LLM理解模块
使用通义千问VL模型理解图片内容
"""

import os
import json
from typing import Dict, Any, List, Optional

# 尝试导入dashscope，如果失败则使用模拟实现
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("未找到dashscope模块，将使用模拟实现")

class MultimodalLLM:
    """多模态大语言模型"""
    
    def __init__(self, api_key: str = None):
        """初始化通义千问VL模型
        
        Args:
            api_key: DashScope API密钥，默认从环境变量读取
        """
        if not DASHSCOPE_AVAILABLE:
            print("⚠️ 多模态LLM初始化失败：dashscope模块未安装")
            return
        
        if api_key:
            dashscope.api_key = api_key
        elif not dashscope.api_key:
            # 尝试从.env文件读取
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if "DASHSCOPE_API_KEY" in line:
                            api_key = line.split("=")[1].strip()
                            dashscope.api_key = api_key
                            break
        
        self.model = "qwen-vl-plus"
        print(f"✅ 多模态LLM初始化完成，使用模型: {self.model}")
    
    def analyze_image(self, 
                     image_path: str, 
                     prompt: str = "请描述这张图片中的商品特点") -> str:
        """分析图片内容
        
        Args:
            image_path: 图片路径或URL
            prompt: 提问提示
            
        Returns:
            图片描述
        """
        if not DASHSCOPE_AVAILABLE:
            return "⚠️ 多模态LLM未安装，无法分析图片"
        
        import base64
        
        # 将图片转换为base64
        def image_to_base64(image_path):
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        # 获取图片扩展名
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        # 使用base64格式
        image_base64 = image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:{mime_type};base64,{image_base64}"},
                    {"text": prompt}
                ]
            }
        ]
        
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get("text", "")
                    if text_content:
                        return text_content
                return str(content)
            else:
                return f"图片分析失败: {response.message}"
        except Exception as e:
            return f"图片分析出错: {str(e)}"
    
    def extract_product_info(self, image_path: str, category: str = None) -> Dict[str, Any]:
        """从图片中提取商品信息
        
        Args:
            image_path: 图片路径
            category: 商品类别（如'fresh'生鲜），用于选择特定的分析prompt
            
        Returns:
            商品信息字典
        """
        if not DASHSCOPE_AVAILABLE:
            return {"error": "多模态LLM未安装，无法分析图片"}
        
        # 根据类别选择不同的prompt
        if category == "fresh" or category == "生鲜":
            # 生鲜专用prompt
            prompt = """请详细分析这张生鲜产品图片，提取以下信息：
1. 商品类型（如水果、蔬菜、肉类、海鲜、乳制品、零食等）
2. 具体品类名称（如苹果、香蕉、西红柿、土豆、牛肉、三文鱼、薯片、饼干、巧克力等）
3. 品牌名称（如苹果的品种、零食的品牌等）
4. 外观特征（颜色、形状、大小、新鲜度、包装等）
5. 品质特点（成熟度、口感、营养价值、口味等）
6. 适用场景（直接食用、烹饪、烘焙、零食、休闲等）
7. 目标用户群体
8. 储存建议（常温、冷藏、冷冻等）
        
请用JSON格式返回，包含以下字段：
- product_type: 商品大类（水果/蔬菜/肉类/海鲜/乳制品/零食等）
- specific_name: 具体品类名称
- brand: 品牌名称（如果能识别）
- appearance_features: 外观特征
- quality_features: 品质特点
- applicable_scenes: 适用场景
- target_users: 目标用户群体
- storage_suggestions: 储存建议"""
        else:
            # 通用商品prompt（包含具体品类）
            prompt = """请详细分析这张商品图片，提取以下信息：
1. 商品类型（如服装、电子产品、美妆护肤品、生鲜食品等）
2. 具体品类名称（如手机、笔记本电脑、口红、面霜、T恤、牛仔裤等）
3. 商品外观特点（颜色、材质、款式等）
4. 适用场景（休闲、正装、运动、办公等）
5. 目标用户群体
6. 风格特点（简约、时尚、复古、新鲜等）
        
请用JSON格式返回，包含以下字段：product_type, specific_name, appearance_features, applicable_scenes, target_users, style_features"""
        
        result = self.analyze_image(image_path, prompt)
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "raw_description": result,
            "product_type": "unknown",
            "error": "无法解析商品信息"
        }
    
    def compare_images(self, 
                       image_paths: List[str], 
                       prompt: str = "请比较这些商品的异同") -> str:
        """比较多个图片
        
        Args:
            image_paths: 图片路径列表
            prompt: 提问提示
            
        Returns:
            比较结果
        """
        if not DASHSCOPE_AVAILABLE:
            return "⚠️ 多模态LLM未安装，无法比较图片"
        
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # 添加图片
        for path in image_paths:
            messages[0]["content"].append(
                {"image": f"file://{os.path.abspath(path)}"}
            )
        
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get("text", "")
                    if text_content:
                        return text_content
                return str(content)
            else:
                return f"图片比较失败: {response.message}"
        except Exception as e:
            return f"图片比较出错: {str(e)}"
    
    def generate_recommendation_based_on_image(self, 
                                              image_path: str,
                                              user_preference: str = "") -> Dict[str, Any]:
        """基于用户上传的图片生成推荐理由
        
        Args:
            image_path: 用户上传的图片路径
            user_preference: 用户偏好描述
            
        Returns:
            推荐信息字典
        """
        if not DASHSCOPE_AVAILABLE:
            return {"error": "多模态LLM未安装，无法分析图片"}
        
        prompt = f"""根据用户上传的图片和以下偏好："{user_preference}"
        
请分析这张图片中的商品风格，并给出推荐理由。
请用JSON格式返回，包含以下字段：
- matched_features: 匹配的特点
- recommendation_reason: 推荐理由
- suggested_keyword: 建议的搜索关键词"""
        
        result = self.analyze_image(image_path, prompt)
        
        try:
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "raw_analysis": result,
            "recommendation_reason": "基于图片分析给出推荐"
        }


class MultimodalRAG:
    """多模态RAG系统"""
    
    def __init__(self):
        """初始化多模态RAG系统"""
        try:
            from modules.multimodal_embedding import MultimodalVectorDB
            from modules.fusion_service import FusionService
            
            # 初始化组件
            self.multimodal_vdb = MultimodalVectorDB(
                collection_name="multimodal_products",
                persist_directory="../models/multimodal_vector_db"
            )
            
            self.llm = MultimodalLLM()
            
            # 传统文本RAG
            self.text_rag = FusionService(category="clothing")
            
            print("✅ 多模态RAG系统初始化完成")
        except Exception as e:
            print(f"⚠️ 多模态RAG系统初始化失败: {e}")
    
    def search(self, 
               query: str = None,
               query_image_path: str = None,
               use_multimodal: bool = True,
               top_k: int = 5) -> List[Dict]:
        """多模态搜索
        
        Args:
            query: 文本查询
            query_image_path: 图片查询路径
            use_multimodal: 是否使用多模态检索
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        results = []
        
        # 1. 文本搜索
        if query:
            try:
                text_results = self.text_rag.query(query, top_k=top_k, strategy="vdb_only")
                results.extend(text_results)
            except Exception as e:
                print(f"文本搜索失败: {e}")
        
        # 2. 多模态搜索
        if use_multimodal and query_image_path:
            try:
                # 先用VL模型分析图片
                image_analysis = self.llm.analyze_image(
                    query_image_path,
                    "请提取这张商品图片的关键特征词"
                )
                
                # 用分析结果进行搜索
                multimodal_results = self.multimodal_vdb.search(
                    query=image_analysis,
                    query_image_path=query_image_path,
                    top_k=top_k
                )
                results.extend(multimodal_results)
            except Exception as e:
                print(f"多模态搜索失败: {e}")
        
        # 去重和排序
        return self._deduplicate_and_rank(results, top_k)
    
    def _deduplicate_and_rank(self, results: List[Dict], top_k: int) -> List[Dict]:
        """去重和排序"""
        seen_ids = set()
        unique_results = []
        
        for r in results:
            pid = r.get("product_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                unique_results.append(r)
        
        # 按相似度排序
        unique_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return unique_results[:top_k]
    
    def chat_with_image(self, 
                        user_input: str,
                        query_image_path: str = None) -> str:
        """基于图片的对话推荐
        
        Args:
            user_input: 用户输入
            query_image_path: 用户上传的图片路径
            
        Returns:
            推荐回复
        """
        context = []
        
        # 如果有图片，先分析图片
        if query_image_path:
            try:
                image_info = self.llm.generate_recommendation_based_on_image(
                    query_image_path,
                    user_input
                )
                context.append(f"用户上传的图片分析: {image_info}")
            except Exception as e:
                print(f"图片分析失败: {e}")
        
        # 搜索相关商品
        try:
            search_results = self.search(
                query=user_input,
                query_image_path=query_image_path,
                top_k=3
            )
            
            # 构建上下文
            if search_results:
                context.append("相关商品:")
                for i, r in enumerate(search_results, 1):
                    context.append(f"{i}. {r.get('text', '')}")
        except Exception as e:
            print(f"搜索失败: {e}")
        
        # 生成回复（这里可以接入LLM生成）
        response = "\n".join(context)
        
        return response


# 测试代码
if __name__ == "__main__":
    # 测试多模态LLM
    print("=== 测试通义千问VL ===")
    llm = MultimodalLLM()
    
    # 测试图片分析（需要实际图片路径）
    test_image = "test.jpg"
    if os.path.exists(test_image):
        result = llm.analyze_image(test_image, "请描述这张图片")
        print(f"图片分析结果: {result}")
    
    # 测试多模态RAG
    print("\n=== 测试多模态RAG ===")
    rag = MultimodalRAG()
    
    # 文本搜索
    results = rag.search(query="夏季T恤")
    print(f"搜索结果: {len(results)} 条")