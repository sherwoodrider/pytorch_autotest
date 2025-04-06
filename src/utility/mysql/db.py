# utils/db.py
import pymysql
from datetime import datetime

sql = """CREATE DATABASE IF NOT EXISTS ml_pipeline;
USE ml_pipeline;
CREATE TABLE IF NOT EXISTS test_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    module VARCHAR(100),
    case_description TEXT,
    result_status VARCHAR(20),
    duration FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);"""



# 根据你本地数据库配置修改
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",  # 请修改为你的数据库密码
    "database": "ml_pipeline",
    "charset": "utf8mb4"
}

def save_result_to_db(module, case_description, result_status, duration=None):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """
            INSERT INTO test_results (module, case_description, result_status, duration, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            module,
            case_description,
            result_status,
            duration,
            datetime.now()
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] 数据保存失败: {e}")
