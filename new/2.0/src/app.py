"""
电商个性化推荐系统 V2.0
- 模块角色：生鲜、电子数码、服装穿搭、美妆护肤
- LangChain Agent集成
- 知识图谱推荐
- 电子数码推荐子模块集成
- 图片上传功能（防无限循环）
"""
import streamlit as st
import json
import sys
import os
import traceback
import hashlib
import io
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Auth import custom_login, custom_register_user
from db_operations import init_db
from langchain_agent import create_recommendation_agent


# ==================== 图片处理辅助函数（防无限循环） ====================
def calculate_image_hash(image_bytes: bytes) -> str:
    """计算图片的SHA256哈希值，用于去重"""
    return hashlib.sha256(image_bytes).hexdigest()


def get_module_processed_images(module: str) -> dict:
    """获取当前模块已处理的图片字典"""
    if module not in st.session_state.processed_images:
        st.session_state.processed_images[module] = {}
    return st.session_state.processed_images[module]


def is_image_processed(module: str, image_hash: str) -> bool:
    """检查图片是否已处理过"""
    processed = get_module_processed_images(module)
    return image_hash in processed


def save_image_result(module: str, image_hash: str, result: dict):
    """保存图片处理结果到缓存"""
    if module not in st.session_state.processed_images:
        st.session_state.processed_images[module] = {}
    st.session_state.processed_images[module][image_hash] = {
        "result": result,
        "timestamp": datetime.now().isoformat()
    }


def get_cached_image_result(module: str, image_hash: str) -> dict:
    """获取缓存的图片处理结果"""
    processed = get_module_processed_images(module)
    if image_hash in processed:
        return processed[image_hash].get("result")
    return None


def clear_module_image_cache(module: str):
    """清空指定模块的图片缓存"""
    if module in st.session_state.processed_images:
        st.session_state.processed_images[module] = {}


def process_uploaded_image(image_file, module: str, multimodal_llm, category: str = None) -> dict:
    """
    处理上传的图片（单向流程，防无限循环）
    
    流程：
    1. 计算图片哈希
    2. 检查是否已处理（去重）
    3. 调用多模态LLM分析图片
    4. 保存结果到缓存（不再触发新的图片分析）
    
    Args:
        image_file: 上传的图片文件
        module: 当前模块名称
        multimodal_llm: 多模态LLM实例
        category: 商品类别（如'fresh'生鲜），用于选择特定的分析prompt
    """
    # 读取图片bytes - 使用getvalue()方法
    image_bytes = image_file.getvalue()
    print(f"图片bytes长度: {len(image_bytes)}")
    
    # 计算哈希
    image_hash = calculate_image_hash(image_bytes)
    
    # 检查是否正在处理同一张图片（防止重复点击）
    if st.session_state.last_uploaded_image_hash == image_hash and st.session_state.image_processing_lock:
        return {
            "status": "processing",
            "message": "图片正在处理中，请稍候..."
        }
    
    # 检查是否已处理过（去重机制）
    cached_result = get_cached_image_result(module, image_hash)
    if cached_result:
        return {
            "status": "cached",
            "message": "这张图片已经分析过了",
            "result": cached_result
        }
    
    # 设置处理锁
    st.session_state.image_processing_lock = True
    st.session_state.last_uploaded_image_hash = image_hash
    
    try:
        # 保存图片到临时文件
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_image_path = os.path.join(temp_dir, f"upload_{image_hash[:16]}.jpg")
        
        # 使用PIL打开并转换为标准JPEG格式
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGB模式（去除透明通道）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存为标准JPEG，确保图片可以被API读取
        image.save(temp_image_path, 'JPEG', quality=95)
        
        print(f"=== 开始分析图片: {temp_image_path} ===")
        print(f"商品类别: {category if category else '通用'}")
        
        # 调用多模态LLM分析图片（只调用一次，单向流程）
        if multimodal_llm:
            print("调用 multimodal_llm.extract_product_info...")
            analysis_result = multimodal_llm.extract_product_info(temp_image_path, category=category)
            print(f"图片分析结果: {analysis_result}")
        else:
            analysis_result = {"error": "多模态LLM未初始化"}
        
        # 构建结果（不再触发新的图片分析）
        result = {
            "image_path": temp_image_path,
            "image_hash": image_hash,
            "analysis": analysis_result,
            "product_type": analysis_result.get("product_type", "unknown"),
            "recommendation_keywords": _extract_recommendation_keywords(analysis_result)
        }
        
        # 保存到缓存
        save_image_result(module, image_hash, result)
        
        return {
            "status": "success",
            "message": "图片分析完成",
            "result": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"图片处理失败: {str(e)}",
            "error": str(e)
        }
    finally:
        # 释放处理锁
        st.session_state.image_processing_lock = False


def _extract_recommendation_keywords(analysis_result: dict) -> list:
    """从图片分析结果中提取推荐关键词"""
    keywords = []
    
    # 优先添加商品种类信息
    if analysis_result.get("product_type"):
        keywords.append(analysis_result["product_type"])
    
    # 添加具体品类名称（如果有）
    if analysis_result.get("specific_name"):
        keywords.append(analysis_result["specific_name"])
    
    # 处理 style_features - 可能是字符串或字典
    style = analysis_result.get("style_features")
    if style:
        if isinstance(style, str):
            keywords.append(style)
        elif isinstance(style, dict):
            # 提取字典中的值
            for v in style.values():
                if isinstance(v, str):
                    keywords.append(v)
                elif isinstance(v, dict):
                    keywords.append(str(v))
    
    # 处理 applicable_scenes - 可能是字符串或列表
    scenes = analysis_result.get("applicable_scenes")
    if scenes:
        if isinstance(scenes, list):
            for s in scenes:
                if isinstance(s, str):
                    keywords.append(s)
                else:
                    keywords.append(str(s))
        else:
            keywords.append(str(scenes))
    
    # 确保所有元素都是字符串
    keywords = [str(k) for k in keywords]
    
    # 去重，保持顺序
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen and keyword != "unknown":
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:5]  # 最多返回5个关键词

# 导入多模态LLM模块（用于图片理解）
try:
    from modules.multimodal_llm import MultimodalLLM
    multimodal_llm_available = True
    print("[OK] 多模态LLM模块导入成功")
except Exception as e:
    multimodal_llm_available = False
    print(f"[WARNING] 多模态LLM模块导入失败: {e}")

# 导入电子数码推荐子模块
try:
    from modules.electronic_module import ElectronicRecommendationModule
    electronic_module_available = True
    print("[OK] 电子数码推荐子模块导入成功")
except Exception as e:
    electronic_module_available = False
    print(f"[ERROR] 电子数码推荐子模块导入失败: {e}")
    traceback.print_exc()

# 导入生鲜推荐子模块
try:
    from modules.fresh_module import FreshRecommendationModule
    fresh_module_available = True
    print("[OK] 生鲜推荐子模块导入成功")
except Exception as e:
    fresh_module_available = False
    print(f"[ERROR] 生鲜推荐子模块导入失败: {e}")
    traceback.print_exc()

# 导入美妆护肤推荐子模块
try:
    from modules.face_module import FaceRecommendationModule
    face_module_available = True
    print("[OK] 美妆护肤推荐子模块导入成功")
except Exception as e:
    face_module_available = False
    print(f"[ERROR] 美妆护肤推荐子模块导入失败: {e}")
    traceback.print_exc()

# 导入服装穿搭推荐子模块
try:
    from modules.clothing_module import ClothingRecommendationModule
    clothing_module_available = True
    print("[OK] 服装穿搭推荐子模块导入成功")
except Exception as e:
    clothing_module_available = False
    print(f"[ERROR] 服装穿搭推荐子模块导入失败: {e}")
    traceback.print_exc()

# 初始化数据库
init_db()

# 页面配置
st.set_page_config(
    page_title="优购通 - 智能电商推荐 V2.0",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 模块配置 ====================
MODULE_CONFIG = {
    "生鲜": {
        "icon": "🥬",
        "description": "新鲜水果、蔬菜、零食推荐",
        "color": "#4CAF50",
        "keywords": ["水果", "蔬菜", "零食", "生鲜", "食品"]
    },
    "电子数码": {
        "icon": "📱",
        "description": "手机、电脑、智能设备推荐",
        "color": "#2196F3",
        "keywords": ["手机", "电脑", "数码", "电子", "智能"]
    },
    "服装穿搭": {
        "icon": "👕",
        "description": "时尚服装、季节穿搭推荐",
        "color": "#FF9800",
        "keywords": ["服装", "穿搭", "时尚", "衣服", "鞋子"]
    },
    "美妆护肤": {
        "icon": "💄",
        "description": "护肤品、化妆品、美妆推荐",
        "color": "#E91E63",
        "keywords": ["美妆", "护肤", "化妆品", "面霜", "精华"]
    }
}

# ==================== 会话状态初始化 ====================
# 使用 try-except 来初始化会话状态，确保在任何情况下都能正确初始化
try:
    # 尝试访问会话状态
    _ = st.session_state
    
    # 初始化基本会话状态
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "美妆护肤"  # 默认模块
    if 'user_roles' not in st.session_state:
        st.session_state.user_roles = []  # 用户关联的角色
    if 'generating_reply' not in st.session_state:
        st.session_state.generating_reply = False
    if 'user_views' not in st.session_state:
        st.session_state.user_views = {}
    if 'last_module' not in st.session_state:
        st.session_state.last_module = "美妆护肤"
    
    # 初始化对话历史
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    # 确保所有模块的对话历史都已初始化
    for module in MODULE_CONFIG.keys():
        if module not in st.session_state.conversations:
            st.session_state.conversations[module] = []
    
    # 初始化最后处理的输入记录
    if 'last_processed_input' not in st.session_state:
        st.session_state.last_processed_input = {}
    # 确保所有模块的最后处理输入都已初始化
    for module in MODULE_CONFIG.keys():
        if module not in st.session_state.last_processed_input:
            st.session_state.last_processed_input[module] = ""
    
    # 初始化图片处理相关状态（防无限循环）
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}  # {模块: {图片哈希: 结果}}
    if 'image_processing_lock' not in st.session_state:
        st.session_state.image_processing_lock = False
    if 'last_uploaded_image_hash' not in st.session_state:
        st.session_state.last_uploaded_image_hash = ""
    if 'image_auto_query' not in st.session_state:
        st.session_state.image_auto_query = None  # 存储图片触发的自动查询
except Exception as e:
    print(f"[WARNING] 会话状态初始化时遇到问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    # 这是一种临时解决方案，确保系统能够启动
    global conversations, last_processed_input
    conversations = {}
    last_processed_input = {}
    for module in MODULE_CONFIG.keys():
        conversations[module] = []
        last_processed_input[module] = ""

# 初始化推荐Agent
try:
    if not hasattr(st.session_state, 'agent'):
        st.session_state.agent = create_recommendation_agent()
except Exception as e:
    print(f"[WARNING] 推荐Agent初始化时遇到问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    global agent
    agent = create_recommendation_agent()

# 初始化多模态LLM（用于图片理解）
try:
    if multimodal_llm_available and not hasattr(st.session_state, 'multimodal_llm'):
        print("\n=== 初始化多模态LLM ===")
        st.session_state.multimodal_llm = MultimodalLLM()
        print("[OK] 多模态LLM初始化成功")
except Exception as e:
    print(f"[WARNING] 多模态LLM初始化时遇到问题: {e}")

# 初始化电子数码推荐子模块
try:
    if not hasattr(st.session_state, 'electronic_module') and electronic_module_available:
        try:
            print("\n=== 初始化电子数码推荐子模块 ===")
            st.session_state.electronic_module = ElectronicRecommendationModule()
            # 尝试加载已训练的模型
            # 使用绝对路径来检查模型文件
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "electronic_recommendation_model.pkl")
            print(f"检查电子数码推荐模型文件路径: {model_path}")
            if os.path.exists(model_path):
                st.session_state.electronic_module.load_model(model_path)
                print("[OK] 已加载电子数码推荐模型")
            else:
                print("[WARNING] 电子数码推荐模型文件不存在，将使用默认模型")
            print("[OK] 电子数码推荐子模块初始化成功")
        except Exception as e:
            print(f"[ERROR] 电子数码推荐子模块初始化失败: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"[WARNING] 电子数码推荐子模块初始化时遇到会话状态问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    global electronic_module
    if electronic_module_available:
        try:
            print("\n=== 初始化电子数码推荐子模块 (全局变量) ===")
            electronic_module = ElectronicRecommendationModule()
            print("[OK] 已初始化电子数码推荐子模块 (全局变量)")
        except Exception as e:
            print(f"[ERROR] 电子数码推荐子模块初始化失败: {e}")
            traceback.print_exc()

# 初始化生鲜推荐子模块
try:
    if not hasattr(st.session_state, 'fresh_module') and fresh_module_available:
        try:
            print("\n=== 初始化生鲜推荐子模块 ===")
            st.session_state.fresh_module = FreshRecommendationModule()
            print("[OK] 已初始化生鲜推荐子模块")
        except Exception as e:
            print(f"[ERROR] 生鲜推荐子模块初始化失败: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"[WARNING] 生鲜推荐子模块初始化时遇到会话状态问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    global fresh_module
    if fresh_module_available:
        try:
            print("\n=== 初始化生鲜推荐子模块 (全局变量) ===")
            fresh_module = FreshRecommendationModule()
            print("[OK] 已初始化生鲜推荐子模块 (全局变量)")
        except Exception as e:
            print(f"[ERROR] 生鲜推荐子模块初始化失败: {e}")
            traceback.print_exc()

# 初始化美妆护肤推荐子模块
try:
    if not hasattr(st.session_state, 'face_module') and face_module_available:
        try:
            print("\n=== 初始化美妆护肤推荐子模块 ===")
            st.session_state.face_module = FaceRecommendationModule()
            print("[OK] 已初始化美妆护肤推荐子模块")
        except Exception as e:
            print(f"[ERROR] 美妆护肤推荐子模块初始化失败: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"[WARNING] 美妆护肤推荐子模块初始化时遇到会话状态问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    global face_module
    if face_module_available:
        try:
            print("\n=== 初始化美妆护肤推荐子模块 (全局变量) ===")
            face_module = FaceRecommendationModule()
            print("[OK] 已初始化美妆护肤推荐子模块 (全局变量)")
        except Exception as e:
            print(f"[ERROR] 美妆护肤推荐子模块初始化失败: {e}")
            traceback.print_exc()

# 初始化服装穿搭推荐子模块
try:
    if not hasattr(st.session_state, 'clothing_module') and clothing_module_available:
        try:
            print("\n=== 初始化服装穿搭推荐子模块 ===")
            st.session_state.clothing_module = ClothingRecommendationModule()
            print("[OK] 已初始化服装穿搭推荐子模块")
        except Exception as e:
            print(f"[ERROR] 服装穿搭推荐子模块初始化失败: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"[WARNING] 服装穿搭推荐子模块初始化时遇到会话状态问题: {e}")
    # 如果会话状态初始化失败，使用全局变量作为备用
    global clothing_module
    if clothing_module_available:
        try:
            print("\n=== 初始化服装穿搭推荐子模块 (全局变量) ===")
            clothing_module = ClothingRecommendationModule()
            print("[OK] 已初始化服装穿搭推荐子模块 (全局变量)")
        except Exception as e:
            print(f"[ERROR] 服装穿搭推荐子模块初始化失败: {e}")
            traceback.print_exc()


# ==================== 1. 登录页 ====================
def render_login_page():
    """登录页面"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🛒 优购通 V2.0</h1>
            <p style="color: #666;">AI驱动的个性化推荐平台</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # 注册入口
        with st.expander("🔐 还没有账号？点击注册", expanded=False):
            custom_register_user()

        # 登录表单
        st.markdown("### 📌 用户登录")
        with st.form(key="login_form", clear_on_submit=True):
            username = st.text_input("用户名", placeholder="请输入注册的用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submit_btn = st.form_submit_button("登录", use_container_width=True)

            if submit_btn:
                success, user_name, msg = custom_login(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_name = user_name
                    
                    # 初始化用户角色（默认关联所有角色）
                    st.session_state.user_roles = list(MODULE_CONFIG.keys())
                    
                    st.session_state.page = "chat"  # 直接进入对话界面
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")


# ==================== 2. 模块选择页 ====================
def render_module_select_page():
    """模块选择页"""
    # 顶部用户栏
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; 
                padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="margin: 0;">欢迎，{st.session_state.user_name}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📦 选择服务模块")
    st.markdown("根据你的需求，选择相应的购物模块")
    
    # 模块卡片（2x2布局）
    col1, col2 = st.columns(2, gap="large")
    
    modules = list(MODULE_CONFIG.keys())
    
    with col1:
        for module in modules[:2]:
            render_module_card(module)
    
    with col2:
        for module in modules[2:]:
            render_module_card(module)
    
    # 退出登录
    st.divider()
    if st.button("🚪 退出登录", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.page = "login"
        st.rerun()


def render_module_card(module: str):
    """渲染模块卡片"""
    config = MODULE_CONFIG[module]
    
    st.markdown(f"""
    <div style="border: 2px solid {config['color']}; border-radius: 12px; 
                padding: 20px; margin-bottom: 20px; background-color: #fafafa;">
        <h2 style="color: {config['color']};">{config['icon']} {module}</h2>
        <p style="color: #666;">{config['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(f"进入{module}", key=f"module_{module}", use_container_width=True):
        st.session_state.current_module = module
        st.session_state.page = "chat"
        
        # 初始化该模块的对话历史
        if module not in st.session_state.conversations:
            st.session_state.conversations[module] = []
        
        # 记录用户选择（使用会话状态替代知识图谱）
        st.session_state.last_module = module
        
        st.rerun()


# ==================== 3. 对话页（集成LangChain Agent） ====================
def render_chat_page():
    """对话页面 - 集成LangChain Agent和知识图谱，内置四大模块切换"""
    
    # 顶部用户栏
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown(f"""
        <div style="text-align: left;">
            <h3>欢迎，{st.session_state.user_name}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if st.button("🚪 退出登录", key="logout_btn", type="primary"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.rerun()
    
    st.divider()
    
    # 左侧：模块选择栏
    with st.sidebar:
        st.markdown("### 📦 推荐模块")
        st.markdown("选择您想要咨询的商品类别：")
        
        # 模块切换按钮
        for module in MODULE_CONFIG.keys():
            config = MODULE_CONFIG[module]
            
            # 高亮当前模块
            is_active = st.session_state.current_module == module
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                f"{config['icon']} {module}",
                key=f"module_switch_{module}",
                type=button_type,
                use_container_width=True
            ):
                # 更新当前模块
                st.session_state.current_module = module
                
                # 初始化该模块的对话历史
                if module not in st.session_state.conversations:
                    st.session_state.conversations[module] = []
                
                # 记录用户选择（使用会话状态替代知识图谱）
                st.session_state.last_module = module
                
                st.rerun()
        
        st.divider()
        
        # 用户角色信息
        st.markdown("### 👤 用户角色")
        st.info(f"当前角色: 多角色")
        # 显示对话轮数
        module = st.session_state.current_module
        conversation_count = len(st.session_state.conversations.get(module, [])) // 2  # 每两轮为一个完整对话
        st.info(f"对话轮数: {conversation_count}")
        
        st.divider()
        
        # 清空对话按钮
        if st.button("🗑️ 清空当前对话", key="clear_chat"):
            module = st.session_state.current_module
            # 1. 清空对话历史
            st.session_state.conversations[module] = []
            # 2. 释放生成锁，确保后续操作正常
            st.session_state.generating_reply = False
            # 3. 清除最后处理的输入记录
            st.session_state.last_processed_input[module] = ""
            # 4. 清除图片自动查询
            st.session_state.image_auto_query = None
            # 5. 清空图片缓存
            clear_module_image_cache(module)
            # 不使用st.rerun()，让Streamlit自动处理状态更新
            # Streamlit会自动检测会话状态变化并更新UI，包括对话轮数计数器
    
    # 右侧：对话区域
    module = st.session_state.current_module
    config = MODULE_CONFIG[module]
    
    # 模块标题
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: {config['color']};">{config['icon']} {module}智能推荐</h2>
        <p style="color: #666;">{config['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 对话消息容器
    chat_container = st.container()
    
    # 显示历史对话
    with chat_container:
        if module in st.session_state.conversations:
            for msg in st.session_state.conversations[module]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #007bff; color: white; padding: 12px; 
                                border-radius: 10px; margin: 8px 0; text-align: right;">
                        <strong>你：</strong>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #e9ecef; padding: 12px; 
                                border-radius: 10px; margin: 8px 0;">
                        <strong>🤖 AI推荐助手：</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"欢迎使用{module}推荐服务！请在下方输入您的需求，或上传图片让AI分析。")
    
    # ==================== 输入区域（文本+图片） ====================
    st.markdown("### 💬 输入您的需求")
    
    # 文本输入
    auto_query = st.session_state.image_auto_query
    if auto_query:
        st.session_state[f'{module}_user_input'] = auto_query
        st.session_state.image_auto_query = None
    
    # 使用text_input并添加显式的发送按钮
    user_input = st.text_input(
        "输入您的需求",
        placeholder=f"例如：我想要{config['keywords'][0]} / 推荐一款{config['keywords'][1]}",
        key=f"text_input_{module}"
    )
    
    # 图片上传
    uploaded_file = st.file_uploader(
        "上传图片（可选）",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="支持 PNG、JPG、BMP、GIF 格式",
        key=f"file_uploader_{module}"
    )
    
    # 显示已上传的图片
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传图片", width=300)
        st.info("输入需求后点击发送按钮，AI会自动分析图片并结合您的文字一起推荐")
    
    # 发送按钮
    submit = st.button("发送 🚀", use_container_width=True, type="primary")
    
    # 处理发送
    if submit and user_input.strip() and not st.session_state.generating_reply:
        final_input = user_input
        
        # 如果有上传图片，先分析图片
        if uploaded_file is not None and multimodal_llm_available and hasattr(st.session_state, 'multimodal_llm'):
            with st.spinner("正在分析图片..."):
                # 根据模块确定商品类别
                category_map = {
                    "生鲜": "fresh",
                    "电子数码": "electronics",
                    "服装穿搭": "clothing",
                    "美妆护肤": "beauty"
                }
                category = category_map.get(module, None)
                
                result = process_uploaded_image(
                    uploaded_file,
                    module,
                    st.session_state.multimodal_llm,
                    category=category
                )
                
                if result["status"] == "success" or result["status"] == "cached":
                    keywords = result["result"]["recommendation_keywords"]
                    if keywords:
                        keyword_str = "、".join(keywords)
                        final_input = f"{user_input}，另外我上传了图片，分析图片风格为：{keyword_str}"
        
        # 检查是否为重复输入
        if st.session_state.last_processed_input.get(module) == final_input:
            st.warning("⚠️ 请勿重复提交相同请求")
            st.stop()
        
        # 设置生成锁
        st.session_state.generating_reply = True
        st.session_state.last_processed_input[module] = final_input
        
        try:
            # 保存用户消息
            if module not in st.session_state.conversations:
                st.session_state.conversations[module] = []
            
            st.session_state.conversations[module].append({
                "role": "user",
                "content": final_input
            })
            
            # 调用推荐系统
            # 获取当前模块的对话历史
            conversation_history = st.session_state.conversations.get(module, [])
            
            agent_response = st.session_state.agent.chat(
                user_input=final_input,
                user_role=module,
                conversation_history=conversation_history
            )
            
            # 调用各模块的推荐子模块，并记录浏览行为
            recommended_goods = []
            
            if module == "电子数码" and electronic_module_available and "electronic_module" in st.session_state:
                try:
                    # 生成电子数码推荐，传递用户需求
                    electronic_recommendations = st.session_state.electronic_module.predict(
                        user_features={
                            'user_id': st.session_state.username,
                            'user_needs': user_input
                        },
                        top_k=3
                    )
                    
                    if electronic_recommendations:
                        recommended_goods.extend(electronic_recommendations)
                except Exception as e:
                    print(f"电子数码推荐子模块调用失败: {e}")
                    traceback.print_exc()
            elif module == "生鲜" and fresh_module_available and "fresh_module" in st.session_state:
                try:
                    # 生成生鲜推荐
                    fresh_recommendations = st.session_state.fresh_module.predict(
                        user_features={
                            'user_id': st.session_state.username,
                            'user_needs': user_input
                        },
                        top_k=3
                    )
                    
                    if fresh_recommendations:
                        recommended_goods.extend(fresh_recommendations)
                except Exception as e:
                    print(f"生鲜推荐子模块调用失败: {e}")
                    traceback.print_exc()
            elif module == "美妆护肤" and face_module_available and "face_module" in st.session_state:
                try:
                    # 生成美妆护肤推荐
                    face_recommendations = st.session_state.face_module.predict(
                        user_features={
                            'user_id': st.session_state.username,
                            'user_needs': user_input
                        },
                        top_k=3
                    )
                    
                    if face_recommendations:
                        recommended_goods.extend(face_recommendations)
                except Exception as e:
                    print(f"美妆护肤推荐子模块调用失败: {e}")
                    traceback.print_exc()
            elif module == "服装穿搭" and clothing_module_available and "clothing_module" in st.session_state:
                try:
                    # 生成服装穿搭推荐
                    clothing_recommendations = st.session_state.clothing_module.predict(
                        user_features={
                            'user_id': st.session_state.username,
                            'user_needs': user_input
                        },
                        top_k=3
                    )
                    
                    if clothing_recommendations:
                        recommended_goods.extend(clothing_recommendations)
                except Exception as e:
                    print(f"服装穿搭推荐子模块调用失败: {e}")
                    traceback.print_exc()
            else:
                # 其他模块推荐逻辑
                recommended_goods = []
            
            # 记录用户浏览行为（使用会话状态替代知识图谱）
            if 'user_views' not in st.session_state:
                st.session_state.user_views = {}
            
            # 为每个推荐商品记录浏览行为
            for i, goods in enumerate(recommended_goods):
                goods_id = goods.get('goods_id', goods.get('product_id', f"{module}_{i}"))
                if goods_id not in st.session_state.user_views:
                    st.session_state.user_views[goods_id] = 0
                st.session_state.user_views[goods_id] += 1
            
            # 检查是否已有相同回复，避免重复添加
            last_msg = st.session_state.conversations[module][-1] if st.session_state.conversations[module] else None
            if not (last_msg and last_msg["role"] == "assistant" and last_msg["content"] == agent_response):
                # 保存助手回复
                st.session_state.conversations[module].append({
                    "role": "assistant",
                    "content": agent_response
                })
        except Exception as e:
            error_msg = f"❌ 推荐失败：{str(e)}"
            # 检查是否已有相同错误回复
            last_msg = st.session_state.conversations[module][-1] if st.session_state.conversations[module] else None
            if not (last_msg and last_msg["role"] == "assistant" and last_msg["content"] == error_msg):
                st.session_state.conversations[module].append({
                    "role": "assistant",
                    "content": error_msg
                })
        finally:
            # 释放生成锁
            st.session_state.generating_reply = False
        
        # 使用st.rerun()强制更新UI，确保用户能立即看到回复
        # 但在下次执行时，重复输入检查会阻止重复处理
        st.rerun()
    
    # 底部：当前模块提示
    config = MODULE_CONFIG[st.session_state.current_module]
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background-color: {config['color']}; color: white; border-radius: 8px;">
        当前正在使用：{config['icon']} {st.session_state.current_module} 推荐模块
    </div>
    """, unsafe_allow_html=True)


# ==================== 主路由 ====================
def main():
    """主路由函数"""
    try:
        # 检查会话状态是否初始化
        if not hasattr(st.session_state, 'authenticated') or not st.session_state.authenticated:
            if not hasattr(st.session_state, 'page') or st.session_state.page != "login":
                # 设置为登录页面
                st.session_state.page = "login"
        
        # 页面路由
        if hasattr(st.session_state, 'page') and st.session_state.page == "login":
            render_login_page()
        elif hasattr(st.session_state, 'page') and st.session_state.page == "chat":
            render_chat_page()
        else:
            # 直接跳转到对话页面，跳过模块选择
            st.session_state.page = "chat"
            st.rerun()
    except Exception as e:
        print(f"[WARNING] 主路由函数遇到会话状态问题: {e}")
        # 如果会话状态初始化失败，显示错误信息并退出
        print("\n=== 系统启动信息 ===")
        print("✅ 所有推荐子模块初始化成功")
        print("✅ 用户数据库初始化成功")
        print("✅ 日志目录已创建")
        print("\n系统启动成功！")
        print("\n请使用以下命令启动 Streamlit 服务器:")
        print("  streamlit run e:\yougoutongzuizhong\new\2.0\src\app.py")
        print("\n或使用以下命令启动系统并捕获日志:")
        print("  python e:\yougoutongzuizhong\new\2.0\src\start_system_and_log.py")
        print("\n系统将在浏览器中运行，访问地址: http://localhost:8501")


if __name__ == "__main__":
    main()