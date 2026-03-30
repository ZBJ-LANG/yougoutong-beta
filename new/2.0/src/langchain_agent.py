"""
LangChain Agent 框架 - 融合通义千问大模型
角色：生鲜、电子数码、服装穿搭、美妆护肤
"""
import os
from typing import Optional, List, Any, Dict

# 尝试导入dashscope，如果失败则使用模拟实现
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("未找到dashscope模块，将使用模拟实现")

# 尝试导入dotenv，如果失败则使用环境变量
try:
    from dotenv import load_dotenv
    # 显式指定.env文件路径，并强制覆盖已存在的环境变量
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path, override=True)
    print(f"已从{env_path}加载配置，DASHSCOPE_API_KEY已设置")
except ImportError:
    print("未找到dotenv模块，将直接使用环境变量")


class QwenLLM:
    """自定义通义千问LLM包装器，简化实现"""
    
    def __init__(self, temperature: float = 0.7):
        """初始化LLM"""
        self.model_name = "qwen-plus"
        self.temperature = temperature
        self.max_tokens = 2000
    
    def invoke(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """调用模型"""
        if not DASHSCOPE_AVAILABLE:
            return "⚠️ 系统提示：dashscope模块未安装，无法调用通义千问模型。请确保已安装dashscope依赖。"
        
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                return "⚠️ 未配置DASHSCOPE_API_KEY环境变量"
            
            dashscope.api_key = api_key
            
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if response.status_code == 200:
                return response.output["text"].strip()
            else:
                return f"⚠️ API调用失败: {response.message}"
        except Exception as e:
            return f"⚠️ 调用异常: {str(e)}"


class RecommendationAgent:
    """推荐Agent - 融合知识图谱和LangChain"""
    
    def __init__(self, kg_query_func=None):
        """
        初始化推荐Agent
        Args:
            kg_query_func: 知识图谱查询函数
        """
        self.llm = QwenLLM(temperature=0.3)
        self.kg_query_func = kg_query_func
        # 添加防止重复调用的机制
        self._last_call_time = 0
        self._call_count = 0
    
    def chat(self, user_input: str, user_role: str = "普通用户", conversation_history: List[Dict] = None) -> str:
        """
        对话接口
        Args:
            user_input: 用户输入
            user_role: 用户角色（生鲜/电子数码/服装穿搭/美妆护肤）
            conversation_history: 对话历史，格式为[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        Returns:
            Agent回复
        """
        import time
        
        # 防止短时间内重复调用（1秒内）
        current_time = time.time()
        if current_time - self._last_call_time < 1:
            self._call_count += 1
            if self._call_count > 3:
                return f"⚠️ 系统繁忙，请稍候再试（已尝试{self._call_count}次）"
            return "请稍候，系统正在处理您的请求..."
        
        self._last_call_time = current_time
        self._call_count = 0
        
        # 获取角色上下文
        role_map = {
            "生鲜": "新鲜水果、蔬菜、零食等食品推荐，注重新鲜度和健康",
            "电子数码": "手机、电脑、智能设备推荐，关注性能参数和技术创新",
            "服装穿搭": "时尚服装、季节穿搭推荐，注重风格和搭配",
            "美妆护肤": "护肤品、化妆品推荐，关注肤质和功效",
            "学生党": "美妆护肤",
            "白领": "服装穿搭",
            "宝妈": "生鲜",
            "科技达人": "电子数码"
        }
        
        # 映射用户角色到模块
        module = role_map.get(user_role, user_role)
        module_context = role_map.get(module, "综合商品推荐")
        
        # 构建对话历史上下文
        history_context = ""
        if conversation_history:
            history_context = "\n\n对话历史：\n"
            for msg in conversation_history:
                content = msg.get("content", "")
                # 确保content是字符串
                if isinstance(content, dict):
                    content = str(content)
                elif not isinstance(content, str):
                    content = str(content)
                if msg.get("role") == "user":
                    history_context += f"用户：{content}\n"
                else:
                    history_context += f"助手：{content}\n"
        
        prompt = f"""
        你是一个电商推荐助手，当前模块是：{module}
        模块特征：{module_context}
        
        {history_context}
        
        用户当前需求：{user_input}
        
        请结合对话历史，分析用户需求，并生成个性化商品推荐。
        请给出推荐结果和理由，格式清晰易读。
        """
        
        return self.llm.invoke(prompt)


def create_recommendation_agent(kg_query_func=None):
    """创建推荐Agent实例"""
    return RecommendationAgent(kg_query_func=kg_query_func)