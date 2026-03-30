import streamlit as st
import hashlib

# 页面配置
st.set_page_config(
    page_title="优购通 - 智能电商推荐系统",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 会话状态初始化
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_module' not in st.session_state:
    st.session_state.current_module = "美妆护肤"
if 'login_error' not in st.session_state:
    st.session_state.login_error = ""
if 'register_success' not in st.session_state:
    st.session_state.register_success = ""

# 密码加密函数
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 模拟用户数据库
USER_DB = {
    "user1": hash_password("password123"),
    "admin": hash_password("admin123")
}

# 登录页面
def render_login_page():
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="color: #2196F3;">优购通 - 智能电商推荐系统</h1>
        <p style="font-size: 18px; color: #666;">请登录以使用推荐服务</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 显示错误信息
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
            st.session_state.login_error = ""
        
        # 显示注册成功信息
        if st.session_state.register_success:
            st.success(st.session_state.register_success)
            st.session_state.register_success = ""
        
        # 登录表单
        st.subheader("登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        
        if st.button("登录"):
            if username in USER_DB and USER_DB[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.page = "main"
                st.rerun()
            else:
                st.session_state.login_error = "用户名或密码错误"
                st.rerun()
        
        # 切换到注册
        st.markdown("---")
        st.subheader("新用户注册")
        new_username = st.text_input("新用户名")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        
        if st.button("注册"):
            if new_username in USER_DB:
                st.session_state.login_error = "用户名已存在"
                st.rerun()
            elif new_password != confirm_password:
                st.session_state.login_error = "两次输入的密码不一致"
                st.rerun()
            elif len(new_password) < 6:
                st.session_state.login_error = "密码长度至少6位"
                st.rerun()
            else:
                USER_DB[new_username] = hash_password(new_password)
                st.session_state.register_success = "注册成功，请登录"
                st.rerun()

# 主页面
def render_main_page():
    # 侧边栏
    st.sidebar.title("模块选择")
    module = st.sidebar.selectbox(
        "选择推荐模块", 
        ["生鲜", "电子数码", "服装穿搭", "美妆护肤"],
        index=["生鲜", "电子数码", "服装穿搭", "美妆护肤"].index(st.session_state.current_module)
    )
    st.session_state.current_module = module
    
    # 主内容
    st.title(f"{module}智能推荐")
    st.write(f"欢迎回来，{st.session_state.username}！")
    st.write("这是一个正在构建中的智能推荐系统。")
    
    # 输入区域
    user_input = st.text_input("输入您的需求")
    if st.button("发送"):
        st.write(f"您的需求：{user_input}")
        st.write("推荐功能正在开发中...")
    
    # 登出按钮
    if st.sidebar.button("登出"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.page = "login"
        st.rerun()

# 页面路由
if not st.session_state.authenticated:
    render_login_page()
else:
    render_main_page()