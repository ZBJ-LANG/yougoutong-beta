#!/bin/bash

# 安装 dashscope 及其依赖
pip install --no-cache-dir dashscope==1.16.0

# 安装其他依赖
pip install --no-cache-dir -r requirements.txt
