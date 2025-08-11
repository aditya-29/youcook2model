import requests
from pathlib import Path
import os

TRAIN_URL = "https://storage.googleapis.com/video_comp/youcook2_comp_train.json"
TEST_URL = "https://storage.googleapis.com/video_comp/youcook2_comp_val.json"


class DownloadData:
    def __init__(self, part="parta", save_folder="./data", file_name = "youcook2_video_comp.json"):
        self.save_folder = Path(save_folder)
        self.dowload_part = part
        self.file_name = Path(file_name)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        if part not in ["parta", "partb", "partc", "partd", "parte", "partf", "partg", "parth", "parti", "partj", "all"]:
            raise Exception("Invalid part")
    

    def run(self,):
        out_path = self.save_folder / self.file_name


        with requests.get(TRAIN_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        f.write(chunk)

        with requests.get(TEST_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        f.write(chunk)


        print("[INFO] Data downloaded successfully")