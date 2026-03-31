#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优购通 - Gradio 版本
"""
import os
import sys
import json
import gradio as gr
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_agent import create_recommendation_agent

try:
    from modules.multimodal_llm import MultimodalLLM
    multimodal_llm_available = True
except:
    multimodal_llm_available = False

agent = create_recommendation_agent()

USER_DATA_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def register(username, password, name):
    users = load_users()
    if username in users:
        return "用户名已存在"
    users[username] = {"password": password, "name": name, "history": {}}
    save_users(users)
    return "注册成功"

def login(username, password):
    users = load_users()
    if username not in users:
        return None, "用户不存在"
    if users[username]["password"] != password:
        return None, "密码错误"
    return users[username]["name"], "登录成功"

session_state = {"conversation_history": {}}

def get_recommendation(user_input, image, module):
    if module not in session_state["conversation_history"]:
        session_state["conversation_history"][module] = []

    enhanced_input = user_input
    image_analysis = None

    if image is not None:
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"upload.png")
        image.save(image_path)

        if multimodal_llm_available:
            try:
                mllm = MultimodalLLM()
                info = mllm.extract_product_info(image_path)
                if isinstance(info, dict):
                    features = []
                    if info.get('product_type'): features.append(info['product_type'])
                    if info.get('style_features'): features.append(info['style_features'])
                    if features:
                        image_analysis = " | ".join(features)
                        enhanced_input = f"{user_input}\n\n[图片分析: {image_analysis}]"
            except:
                pass

    session_state["conversation_history"][module].append({"role": "user", "content": enhanced_input})
    response = agent.chat(user_input=enhanced_input, user_role=module, conversation_history=session_state["conversation_history"][module])
    session_state["conversation_history"][module].append({"role": "assistant", "content": response})

    result = f"**推荐结果:**\n\n{response}"
    if image_analysis:
        result = f"**图片识别:** {image_analysis}\n\n" + result

    return result

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', sans-serif !important;
    background: #FFFFFF !important;
    min-height: 100vh;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.main {
    margin: 0 !important;
    padding: 0 !important;
}

/* Auth Page */
.auth-wrapper {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #FFFFFF;
}

.auth-box {
    width: 100%;
    max-width: 380px;
    padding: 3rem 2.5rem;
}

.auth-logo {
    text-align: center;
    margin-bottom: 2.5rem;
}

.auth-logo h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #111111;
    letter-spacing: 0.05em;
}

.auth-logo p {
    font-size: 0.85rem;
    color: #999999;
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
}

.auth-input {
    width: 100%;
    padding: 0.875rem 1rem;
    border: 1px solid #E5E5E5 !important;
    border-radius: 8px !important;
    background: #FAFAFA !important;
    font-size: 0.95rem !important;
    color: #111111 !important;
    margin-bottom: 1rem;
    transition: all 0.2s ease !important;
}

.auth-input:focus {
    border-color: #111111 !important;
    background: #FFFFFF !important;
    outline: none !important;
    box-shadow: none !important;
}

.auth-input::placeholder { color: #CCCCCC !important; }

.auth-btn {
    width: 100%;
    padding: 0.875rem;
    background: #111111 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem;
}

.auth-btn:hover { background: #333333 !important; }

.auth-error {
    color: #DC2626;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 0.75rem;
    padding: 0.75rem;
    background: #FEF2F2;
    border-radius: 8px;
}

.auth-success {
    color: #16A34A;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 0.75rem;
    padding: 0.75rem;
    background: #F0FDF4;
    border-radius: 8px;
}

/* Main Page */
.main-wrapper {
    min-height: 100vh;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding: 0;
    width: 100%;
}

.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 3rem;
    border-bottom: 1px solid #F0F0F0;
    background: #FFFFFF;
}

.main-header .logo {
    font-size: 1.25rem;
    font-weight: 600;
    color: #111111;
    letter-spacing: 0.05em;
}

.main-header .user-area {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.main-header .user-name {
    font-size: 0.9rem;
    color: #666666;
}

.main-header .user-name strong {
    color: #111111;
    font-weight: 500;
}

.main-header .logout-btn {
    padding: 0.5rem 1rem;
    background: transparent !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 6px !important;
    color: #666666 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.main-header .logout-btn:hover {
    border-color: #111111 !important;
    color: #111111 !important;
}

.main-content {
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem;
}

.content-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 2rem;
}

.sidebar {
    background: #FAFAFA;
    border-radius: 12px;
    padding: 1.5rem;
}

.sidebar-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

.module-option {
    display: block;
    width: 100%;
    padding: 0.75rem 1rem;
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    color: #333333;
    font-size: 0.9rem;
    text-align: left;
    cursor: pointer;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}

.module-option:hover { border-color: #111111; }
.module-option.selected { background: #111111; border-color: #111111; color: #FFFFFF; }

.action-btn {
    width: 100%;
    padding: 0.75rem;
    background: transparent !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 8px !important;
    color: #666666 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    margin-top: 0.75rem;
    transition: all 0.2s ease !important;
}

.action-btn:hover { border-color: #111111 !important; color: #111111 !important; }

.main-card {
    background: #FAFAFA;
    border-radius: 12px;
    padding: 1.5rem;
}

.card-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

.upload-box {
    border: 2px dashed #E5E5E5 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    min-height: 140px;
    margin-bottom: 1rem;
}

.input-box {
    width: 100%;
    padding: 0.875rem 1rem;
    border: 1px solid #E5E5E5 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    font-size: 0.95rem !important;
    color: #111111 !important;
    margin-bottom: 1rem;
    transition: all 0.2s ease !important;
}

.input-box:focus {
    border-color: #111111 !important;
    outline: none !important;
    box-shadow: none !important;
}

.submit-btn {
    width: 100%;
    padding: 0.875rem;
    background: #111111 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.submit-btn:hover { background: #333333 !important; }

.result-box {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 200px;
    margin-top: 1.5rem;
}

.result-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

.result-content { color: #333333; line-height: 1.7; font-size: 0.95rem; }
.empty-state { color: #CCCCCC; text-align: center; padding: 2rem; font-style: italic; }

#auth-col > .primary { display: none !important; }
"""

with gr.Blocks(title="优购通", css=custom_css) as demo:
    with gr.Column(visible=True, elem_classes="auth-wrapper", elem_id="auth-col") as auth_col:
        with gr.Group(elem_classes="auth-box"):
            gr.HTML("""
                <div class="auth-logo">
                    <h1>优购通</h1>
                    <p>智能推荐平台</p>
                </div>
            """)

            with gr.Tab("登录"):
                login_user = gr.Textbox(placeholder="用户名", elem_classes="auth-input", label="")
                login_pass = gr.Textbox(placeholder="密码", type="password", elem_classes="auth-input", label="")
                login_btn = gr.Button("登录", elem_classes="auth-btn")
                login_msg = gr.HTML("")

            with gr.Tab("注册"):
                reg_user = gr.Textbox(placeholder="用户名", elem_classes="auth-input", label="")
                reg_pass = gr.Textbox(placeholder="密码", type="password", elem_classes="auth-input", label="")
                reg_name = gr.Textbox(placeholder="昵称", elem_classes="auth-input", label="")
                reg_btn = gr.Button("注册", elem_classes="auth-btn")
                reg_msg = gr.HTML("")

    with gr.Column(visible=False, elem_classes="main-wrapper", elem_id="main-col") as main_col:
        user_display = gr.HTML("""
            <div class="main-header">
                <div class="logo">优购通</div>
                <div class="user-area">
                    <span class="user-name">欢迎，<strong>用户</strong></span>
                    <button class="logout-btn">退出</button>
                </div>
            </div>
        """)

        gr.HTML("""
            <div class="main-content">
                <div class="content-grid">
                    <div class="sidebar">
                        <div class="sidebar-title">商品模块</div>
        """)

        module_radio = gr.Radio(["生鲜", "电子数码", "服装穿搭", "美妆护肤"], value="美妆护肤")

        gr.HTML("""
                        </div>
                    </div>
                    <div class="main-card">
                        <div class="card-header">描述需求</div>
        """)

        image_input = gr.Image(label="", type="pil", height=140)
        user_input = gr.Textbox(placeholder="描述您想要的商品...", lines=3, elem_classes="input-box", label="")
        submit_btn = gr.Button("获取推荐", elem_classes="submit-btn")

        gr.HTML("""
                    </div>
                </div>
                <div class="result-box">
                    <div class="result-title">推荐结果</div>
                    <div class="result-content">
                        <div class="empty-state">推荐结果将显示在这里</div>
                    </div>
                </div>
            </div>
        """)

        output = gr.HTML("")

    def do_login(u, p):
        name, msg = login(u, p)
        if name:
            welcome_html = f"""
            <div class="main-header">
                <div class="logo">优购通</div>
                <div class="user-area">
                    <span class="user-name">欢迎，<strong>{name}</strong></span>
                </div>
            </div>
            """
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                welcome_html
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f'<div class="auth-error">{msg}</div>'
        )

    def do_reg(u, p, n):
        msg = register(u, p, n)
        if "成功" in msg:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                '<div class="auth-success">注册成功，请登录</div>'
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f'<div class="auth-error">{msg}</div>'
        )

    login_btn.click(do_login, [login_user, login_pass], [auth_col, main_col, user_display])
    reg_btn.click(do_reg, [reg_user, reg_pass, reg_name], [auth_col, main_col, user_display])

    def do_submit(txt, img, mod):
        if not txt.strip():
            return "请输入需求"
        return get_recommendation(txt, img, mod)

    submit_btn.click(do_submit, [user_input, image_input, module_radio], [output])

    def do_logout():
        session_state["conversation_history"] = {}
        return (
            gr.update(visible=True),
            gr.update(visible=False)
        )

    logout_btn = gr.Button("退出", visible=False)
    logout_btn.click(do_logout, [], [auth_col, main_col])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863)
