import streamlit as st
import bcrypt
from db_operations import get_user_by_username, register_user_to_db, verify_password


# 自定义登录验证（数据库版）
def custom_login(username: str, password: str):
    if not username or not password:
        return False, None, "用户名/密码不能为空"

    user = get_user_by_username(username)
    if not user:
        return False, None, "用户名不存在"

    if verify_password(password, user["password_hash"]):
        return True, user["name"], "登录成功"
    else:
        return False, None, "密码错误"


# 自定义注册表单（适配前端界面）
def custom_register_user():
    with st.form(key="register_form", clear_on_submit=True):
        st.subheader("📝 新用户注册")
        username = st.text_input("用户名", placeholder="请输入唯一用户名", key="reg_username")
        name = st.text_input("昵称", placeholder="请输入显示名称", key="reg_name")
        email = st.text_input("邮箱（可选）", placeholder="请输入邮箱地址", key="reg_email")
        password = st.text_input("密码", type="password", placeholder="请输入密码", key="reg_pwd")
        confirm_pwd = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_confirm_pwd")

        submit_btn = st.form_submit_button("提交注册")
        if submit_btn:
            if password != confirm_pwd:
                st.error("❌ 两次密码输入不一致")
            elif len(password) < 6:
                st.error("❌ 密码长度不能少于6位")
            else:
                # 调用db_operations的注册函数（已正确导入）
                success, msg = register_user_to_db(username, password, name, email)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")