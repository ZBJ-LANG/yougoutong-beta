import sqlite3
from passlib.hash import bcrypt
import datetime
import os

# 数据库文件路径
DB_PATH = "ecommerce_users.db"


# 🔴 确保hash_password函数在register_user_to_db之前定义
def hash_password(password: str) -> str:
    """密码哈希（bcrypt，不可逆）"""
    return bcrypt.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """验证密码（对比哈希值）"""
    return bcrypt.verify(password, hashed_password)


def init_db():
    """初始化用户数据库（创建user表）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    print("[OK] 用户数据库初始化成功")


def register_user_to_db(username: str, password: str, name: str, email: str = None):
    """注册用户到数据库（调用上面定义的hash_password）"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 检查用户名是否已存在
        cursor.execute("SELECT * FROM user WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "用户名已存在"

        # 🔴 此处调用hash_password（已在上方定义，可正常访问）
        password_hash = hash_password(password)
        cursor.execute('''
        INSERT INTO user (username, password_hash, name, email)
        VALUES (?, ?, ?, ?)
        ''', (username, password_hash, name, email))

        conn.commit()
        conn.close()
        return True, "注册成功！请返回登录"
    except Exception as e:
        return False, f"注册失败：{str(e)}"


def get_user_by_username(username: str):
    """根据用户名查询用户信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT username, password_hash, name, email FROM user WHERE username = ?
    ''', (username,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return {
            "username": user[0],
            "password_hash": user[1],
            "name": user[2],
            "email": user[3]
        }
    return None


# 首次运行自动初始化数据库
if not os.path.exists(DB_PATH):
    init_db()