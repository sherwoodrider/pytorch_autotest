
from datasets import load_dataset
import pandas as pd
import os

class DownloadDataset():
    def __init__(self,dataset_name):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self.dataset_name = dataset_name
        self.train_limit = 50
        self.test_limit = 10
        self.cache_dir = "D:\code_repo\pytorch_autotest\data\csv_output"
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
    def download(self):
        print("download {} begin".format(self.dataset_name))
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        print("download dataset :", self.dataset)

    def arrow_to_csv(self,arrow_path):
        df = pd.read_feather(arrow_path)
        csv_path = arrow_path.replace(".arrow",".csv")
        df.to_csv(csv_path, index=False)

    def download_train_data_to_csv(self):
        print("download_train_data_to_csv {} begin".format(self.dataset_name))
        # self.train_dataset = load_dataset(self.dataset_name ,split="train", cache_dir=self.cache_dir) # 下载源数据集
        self.train_dataset = load_dataset(self.dataset_name, split="train")
        df = pd.DataFrame(self.train_dataset)
        # 打乱数据
        shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)
        top_data = shuffled_data.head(self.train_limit)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        train_save_path = os.path.join(self.cache_dir, "{}_train.csv".format(self.dataset_name))
        top_data.to_csv(train_save_path, index=False)
        print("{} is already save to {}".format(self.dataset_name,train_save_path))

    def download_test_data_to_csv(self):
        print("download_test_data_to_csv {} begin".format(self.dataset_name))
        self.test_dataset = load_dataset(self.dataset_name, split="test")
        df = pd.DataFrame(self.test_dataset)
        # 打乱数据
        shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)
        top_data = shuffled_data.head(self.test_limit)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        save_path = os.path.join(self.cache_dir, "{}_test.csv".format(self.dataset_name))
        top_data.to_csv(save_path, index=False)
        print("{} is already save to {}".format(self.dataset_name,save_path))

    def translate_ag_news_label(self):
        label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }

    def translate_imdb_label(self):
        label_map = {
            0: "negative",
            1: "positive",
        }


if __name__ == '__main__':
    # name = "imdb"
    name = "ag_news"
    d = DownloadDataset(name)
    d.download_train_data_to_csv()
    d.download_test_data_to_csv()

