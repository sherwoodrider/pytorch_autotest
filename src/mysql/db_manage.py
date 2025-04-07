# test_results/mysql_logger.py

import pymysql
from datetime import datetime

class MySQLTestLogger:
    def __init__(self, host="localhost", port=3306, user="root", password="", database="test_results", table="test_logs"):
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4"
        )
        self.table = table
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        with self.connection.cursor() as cursor:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{self.table}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                test_name VARCHAR(255),
                category VARCHAR(64),
                status VARCHAR(10),
                duration FLOAT,
                message TEXT,
                created_at DATETIME
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_sql)
            self.connection.commit()

    def log_result(self, test_name, category, status, duration, message=""):
        with self.connection.cursor() as cursor:
            insert_sql = f"""
            INSERT INTO `{self.table}` (test_name, category, status, duration, message, created_at)
            VALUES (%s, %s, %s, %s, %s, %s);
            """
            cursor.execute(insert_sql, (
                test_name, category, status, duration, message, datetime.now()
            ))
            self.connection.commit()

    def close(self):
        self.connection.close()
