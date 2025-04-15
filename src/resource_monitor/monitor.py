import multiprocessing
import psutil
import time
from datetime import datetime

class ResourceMonitor:
    def __init__(self, log_file='cpu_monitor.log'):
        self.log_file = log_file
        self.queue = multiprocessing.Queue()
        self._process = None

    def _monitor_loop(self):
        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                cpu_percent = psutil.cpu_percent(interval=1)
                self.queue.put((timestamp, cpu_percent))
                self._write_log(timestamp, cpu_percent)

            except KeyboardInterrupt:
                break

    def _write_log(self, timestamp, cpu_percent):
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - CPU Usage: {cpu_percent}%\n")

    def start(self):
        if not self._process or not self._process.is_alive():
            self._process = multiprocessing.Process(
                target=self._monitor_loop,
                daemon=True
            )
            self._process.start()

    def stop(self):
        if self._process and self._process.is_alive():
            self._process.terminate()

    def get_latest_usage(self):
        latest = None
        while not self.queue.empty():
            latest = self.queue.get()
        return latest[1] if latest else 0  # 返回百分比数值


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()