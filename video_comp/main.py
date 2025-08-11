import pandas as pd
import argparse
from pathlib import Path
import os

from download_data import DownloadData
from create_chunks import CreateChunks
from train_nn import TrainNN

# ----- GLOBAL -----
MAX_WORKERS = min(os.cpu_count() or 4, 32)              # reasonable default

# ----- DOWNLOAD DATA -----
DATA_PATH = Path("/mnt/localssd/video_comp")
DATA_FILE_NAME = "youcook2_video_comp.json"

# ----- CREATE CHUNKS -----
RAW_VIDEO_DIR = Path("/mnt/localssd/raw_videos")
CHUNK_VIDEO_DIR = Path("/mnt/localssd/video_comp/chunk_videos")

# ----- TRAIN -----
FPS = 5
MAX_FRAMES_PER_VIDEO = None
BATCH_SIZE_EMBED = 64
BATCH_SIZE_TRAIN = 256
NUM_EPOCHS = 3
LR = 1E-3


print("MAX WORKERS :: ", MAX_WORKERS)



class CreateData:
    def __init__(self, ):
        pass


    def parse_args(self,):
        parser = argparse.ArgumentParser(
            description = "video comp processing"
        )

        parser.add_argument(
            "--download_part",
            help="download part"
        )

        parser.add_argument(
            "--max_videos",
            help="max videos",
            default=3
        )


        return parser.parse_args()


    def main(self,
             download_part = "parta",
             max_videos = None):
        
        # download data
        DownloadData(part = download_part, 
                     save_folder=DATA_PATH,
                     file_name = DATA_FILE_NAME).run()

        # create chunks
        CreateChunks(video_directory = RAW_VIDEO_DIR,
                    out_path = CHUNK_VIDEO_DIR,
                    json_path = DATA_PATH / DATA_FILE_NAME,
                    cpu_count = MAX_WORKERS).run()
        
        # train model
        TrainNN().train()
        
        
if __name__ == "__main__":
    C = CreateData()
    args = C.parse_args()

    C.main(
        download_part=args.download_part,
        max_videos=args.max_videos
    )
