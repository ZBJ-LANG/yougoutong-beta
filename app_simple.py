import streamlit as st

# 简单的Streamlit应用
st.title("优购通 - 智能电商推荐系统")
st.write("这是一个最小化的部署版本")

# 测试基本功能
st.sidebar.title("模块选择")
module = st.sidebar.selectbox("选择推荐模块", ["生鲜", "电子数码", "服装穿搭", "美妆护肤"])

st.write(f"您选择的模块是：{module}")
st.write("应用已成功部署！")