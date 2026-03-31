#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动系统并将日志保存到文件
"""

import os
import sys
import subprocess
import time

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== 启动系统并将日志保存到文件 ===")
print("=" * 60)

# 定义日志文件路径
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_startup.log")
print(f"日志将保存到: {log_file_path}")

# 启动2.0版本的系统并捕获日志
def start_system_and_log():
    """启动2.0版本的系统并捕获日志"""
    print("\n启动2.0版本的系统...")
    # 启动2.0版本的系统
    process = subprocess.Popen(
        ["python", "app.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    # 打开日志文件
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        # 写入启动时间
        log_file.write(f"系统启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n")
        
        # 捕获输出
        stdout = []
        stderr = []
        
        # 等待系统启动，最多等待60秒
        start_time = time.time()
        while time.time() - start_time < 60:
            line = process.stdout.readline()
            if line:
                stdout.append(line)
                log_file.write(line)
                print(line.strip())
            
            error_line = process.stderr.readline()
            if error_line:
                stderr.append(error_line)
                log_file.write(f"错误: {error_line}")
                print(f"错误: {error_line.strip()}")
            
            # 检查进程是否已经退出
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # 读取剩余的输出
        remaining_stdout = process.stdout.read()
        if remaining_stdout:
            stdout.append(remaining_stdout)
            log_file.write(remaining_stdout)
            print(remaining_stdout.strip())
        
        remaining_stderr = process.stderr.read()
        if remaining_stderr:
            stderr.append(remaining_stderr)
            log_file.write(f"错误: {remaining_stderr}")
            print(f"错误: {remaining_stderr.strip()}")
        
        # 写入结束时间
        log_file.write("=" * 80 + "\n")
        log_file.write(f"系统启动结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return stdout, stderr

if __name__ == "__main__":
    # 启动系统并将日志保存到文件
    stdout, stderr = start_system_and_log()
    
    print("\n=== 系统启动完成 ===")
    print(f"✅ 系统启动日志已保存到: {log_file_path}")
    print("\n请查看日志文件以获取完整的系统启动详情。")
    print("\n=== 测试完成 ===")
