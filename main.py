import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from download_data import DownloadData
from create_chunks import CreateChunk
from apply_decorators import ApplyDecorators
from create_frames_gpu import create_frames_gpu

DATA_PATH = Path("./data")

# ----------- CREATE CHUNKS -----------
RAW_VIDEO_ROOT   = DATA_PATH / Path("./raw_videos")        # original variable from your code
RAW_ANNOT_ROOT   = DATA_PATH / Path("./raw_annot_videos")
EXTENSIONS       = [".mp4", ".mkv", ".mov"]         # whatever you support
SECONDS_LIMIT    = 50                              # your existing guardrail
MAX_WORKERS      = os.cpu_count() or 4              # reasonable default
# -------------------------------------

# ----------- APPLY DECORATORS -----------
SAVE_ANNOT_ROOT     = DATA_PATH / Path("./annot_videos")
CAPTION_FILE  = DATA_PATH / Path("captions.json")
# ----------------------------------------

# ----------- CREATE FRAMES -----------
SAVE_FRAMES_ROOT = DATA_PATH / Path("./annot_frames")
FPS = 2.0
VERIFY = True
MAX_VIDEOS = 10
# -------------------------------------

print("MAX WORKERS : ", MAX_WORKERS)

class CreateData:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    def extract_labels_annotations(self,):
        # extract labels
        label_foodtype_mp = {}
        path = os.path.join(self.data_path, "label_foodtype.csv")

        df = pd.read_csv(path, header=None)
        for iter, row in df.iterrows():
            label_foodtype_mp[row[0]] = row[1]

        # extract annotations
        train_val_annot_path = os.path.join(self.data_path, "youcookii_annotations_trainval.json")
        test_annot_path = os.path.join(self.data_path, "youcookii_annotations_test_segments_only.json")

        with open(train_val_annot_path, 'r') as f:
            train_val_annot_mp = json.load(f)


        with open(test_annot_path, 'r') as f:
            test_annot_mp = json.load(f)

        # combine train_val_annot_mp and test_annot_mp together
        for key in test_annot_mp["database"]:
            if key in train_val_annot_mp["database"]:
                print("key already present : ", key)
                continue

            train_val_annot_mp["database"][key] = test_annot_mp["database"][key]
        self.annot_mp = train_val_annot_mp.copy()

        # print stats
        print("total unique food types : ", len(label_foodtype_mp))
        print("len of train_val_annot_mp : ", len(train_val_annot_mp["database"]))
        print("len of test_annot_path : ", len(test_annot_mp["database"]))
        print("len of annot_mp : ", len(self.annot_mp["database"]))

    def annalyze_annotations(self):
        self.video_annotations = {}
        total_annotations = 0
        annotation_length = []
        __outlier_annotation = []

        empty_captions = 0

        # number of unique chunks
        for video in self.annot_mp["database"]:
            annotations = self.annot_mp["database"][video]

            if video not in self.video_annotations:
                self.video_annotations[video] = []

            for annotation in annotations["annotations"]:
                temp = {}
                temp["start"] = annotation["segment"][0]
                temp["end"] = annotation["segment"][1]
                temp["id"] = annotation["id"]
                temp["sentence"] = annotation["sentence"]
                temp["video"] = video
                temp["subset"] = self.annot_mp["database"][video]["subset"]
                temp["recipe_type"] = self.annot_mp["database"][video]["recipe_type"]
                self.video_annotations[video].append(temp)
    
                # stats
                total_annotations += 1
                length = temp["end"] - temp["start"]
                annotation_length.append(length)
                if temp["sentence"].strip() == "":
                    empty_captions += 1
    
                if length > 50:
                    __outlier_annotation.append(temp)
    

        print("*** Across YouCook2 Dataset ***")
        print("total annotations : ", total_annotations)
        print("average annotations length (seconds) : ", sum(annotation_length) / len(annotation_length))
        print("max annotations length (seconds): ", max(annotation_length))
        print("min annotations length (seconds): ", min(annotation_length))
        print("total empty captions : ", empty_captions)
        print("total annotations less than 50 seconds : ", sum(1 for x in annotation_length if x <= 50))

        print(__outlier_annotation[0])

        # print the quartile plot for annotation length
        plt.figure(figsize=(6, 4))
        plt.boxplot(annotation_length, vert=True, showfliers=True, patch_artist=True)
        plt.ylabel("Seconds")
        plt.title("Quartile plot of annotation lengths")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        # plt.show()
        plt.savefig("./annotations_box_plot.png")

    def _create_chunks_driver(self):
        CreateChunk(video_annotations=self.video_annotations,
                     raw_video_root=RAW_VIDEO_ROOT,
                     raw_annot_root=RAW_ANNOT_ROOT,
                     extensions=EXTENSIONS,
                     seconds_limit=SECONDS_LIMIT,
                     max_workers=MAX_WORKERS,
                     data_folder_path=DATA_PATH).run()
        
    def _apply_decorators(self):
        ApplyDecorators(raw_annot_root=RAW_ANNOT_ROOT,
                        save_annot_root=SAVE_ANNOT_ROOT,
                        caption_file_path=CAPTION_FILE,
                        cpu_count=MAX_WORKERS,
                       video_annotations=self.video_annotations).run()

    def _extract_frames(self):
        create_frames_gpu(input_root=RAW_ANNOT_ROOT,
                      output_root=SAVE_FRAMES_ROOT,
                      fps=FPS,
                      max_videos=MAX_VIDEOS,
                      verify=VERIFY)


    def __str2bool(self, v: str) -> bool:
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")


    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Demo: optionally skip the data‑download step."
        )
    
        parser.add_argument(
            "--skip_download_data",
            type=self.__str2bool,
            nargs="?",
            const=True,          # allows `--skip_download_data` (no value) → True
            default=False,
            help="Skip the data‑download step (true/false)."
        )

        parser.add_argument(
            "--skip_create_chunks",
            type=self.__str2bool,
            nargs="?",
            const=True,          # allows `--skip_download_data` (no value) → True
            default=False,
            help="Skip the chunk creation step (true/false)."
        )

        parser.add_argument(
            "--skip_apply_decorators",
            type=self.__str2bool,
            nargs="?",
            const=True,          # allows `--skip_download_data` (no value) → True
            default=False,
            help="Skip the apply decorators step (true/false)."
        )

        parser.add_argument(
            "--skip_extract_frames",
            type=self.__str2bool,
            nargs="?",
            const=True,          # allows `--skip_download_data` (no value) → True
            default=False,
            help="Skip the extract frames step (true/false)."
        )

        parser.add_argument(
            "--download_part",
            help="download part"
        )
            
            
        return parser.parse_args()


    def main(self, 
             download_part="parta",
             skip_download_data=False,
             skip_create_chunks=False,
             skip_apply_decorators=False,
             skip_extract_frames=False):
        
        if not skip_download_data:
            # STEP 1: download the data into disk
            DownloadData(part=download_part, save_folder=DATA_PATH).run()

        # STEP 2: Download the labels and annotations
        self.extract_labels_annotations()

        # STEP 3: analyze the annotations
        self.annalyze_annotations()

        # STEP 4: Create chunks from videos
        if not skip_create_chunks:
            self._create_chunks_driver()

        # STEP 5: Apply Decorators
        if not skip_apply_decorators:
            self._apply_decorators()

        # STEP 6: Save as frames
        if not skip_extract_frames:
            self._extract_frames()

        

if __name__ == "__main__":
    C = CreateData()
    args = C.parse_args()
    C.main(download_part = args.download_part,
           skip_download_data = args.skip_download_data,
           skip_create_chunks = args.skip_create_chunks,
           skip_apply_decorators = args.skip_apply_decorators,
           skio_extract_frames = args.skip_extract_frames)
        


        




