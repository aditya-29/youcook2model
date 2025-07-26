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

DATA_PATH = Path("/mnt/localssd")

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

# New configuration for decorators
ENABLE_REVERSE = True
ENABLE_SPEED_CHANGE = True  
ENABLE_TEMPORAL_ORDER = True
SPEED_FACTORS = [0.5, 1.0, 2.0]  # Configure which speed factors to use
# ----------------------------------------

# ----------- CREATE FRAMES -----------
SAVE_FRAMES_ROOT = DATA_PATH / Path("./annot_frames")
FPS = 2.0
VERIFY = True
# -------------------------------------

print("MAX WORKERS : ", MAX_WORKERS)

class CreateData:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    def extract_labels_annotations(self,):
        # extract labels
        label_foodtype_mp = {}
        path = os.path.join(self.data_path, "label_foodtype.csv")

        try:
            df = pd.read_csv(path, header=None)
            for iter, row in df.iterrows():
                label_foodtype_mp[row[0]] = row[1]
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping label extraction...")
            return

        # extract annotations
        train_val_annot_path = os.path.join(self.data_path, "youcookii_annotations_trainval.json")
        test_annot_path = os.path.join(self.data_path, "youcookii_annotations_test_segments_only.json")

        try:
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
            
        except FileNotFoundError as e:
            print(f"Warning: Annotation file not found: {e}")
            self.annot_mp = {"database": {}}

    def analyze_annotations(self):  # Fixed typo from "annalyze"
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
    
        if not annotation_length:
            print("*** No annotations found in dataset ***")
            return

        print("*** Across YouCook2 Dataset ***")
        print("total annotations : ", total_annotations)
        print("average annotations length (seconds) : ", sum(annotation_length) / len(annotation_length))
        print("max annotations length (seconds): ", max(annotation_length))
        print("min annotations length (seconds): ", min(annotation_length))
        print("total empty captions : ", empty_captions)
        print("total annotations less than 50 seconds : ", sum(1 for x in annotation_length if x <= 50))

        if __outlier_annotation:
            print("Sample outlier annotation:", __outlier_annotation[0])

        # print the quartile plot for annotation length
        plt.figure(figsize=(6, 4))
        plt.boxplot(annotation_length, vert=True, showfliers=True, patch_artist=True)
        plt.ylabel("Seconds")
        plt.title("Quartile plot of annotation lengths")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("./annotations_box_plot.png")
        print("Saved annotation analysis plot to: ./annotations_box_plot.png")

    def _create_chunks_driver(self, max_videos):
        print(f"Creating chunks with max_videos: {max_videos}")
        CreateChunk(video_annotations=self.video_annotations,
                     raw_video_root=RAW_VIDEO_ROOT,
                     raw_annot_root=RAW_ANNOT_ROOT,
                     extensions=EXTENSIONS,
                     seconds_limit=SECONDS_LIMIT,
                     max_workers=MAX_WORKERS,
                     data_folder_path=DATA_PATH).run(max_videos=max_videos)
        
    def _apply_decorators(self):
        print("Applying video decorators (reverse, speed, temporal order)...")
        print(f"Enabled decorators:")
        print(f"  - Reverse: {ENABLE_REVERSE}")
        print(f"  - Speed Change: {ENABLE_SPEED_CHANGE} (factors: {SPEED_FACTORS})")
        print(f"  - Temporal Order: {ENABLE_TEMPORAL_ORDER}")
        
        ApplyDecorators(raw_annot_root=RAW_ANNOT_ROOT,
                        save_annot_root=SAVE_ANNOT_ROOT,
                        caption_file_path=CAPTION_FILE,
                        cpu_count=MAX_WORKERS,
                       video_annotations=self.video_annotations).run()
        
        print(f"Decorator processing complete. Results saved to: {SAVE_ANNOT_ROOT}")
        print(f"Caption mappings saved to: {CAPTION_FILE}")

    def _extract_frames(self):
        print(f"Extracting frames at {FPS} FPS...")
        create_frames_gpu(input_root=SAVE_ANNOT_ROOT,
                      output_root=SAVE_FRAMES_ROOT,
                      fps=FPS,
                      max_videos=None,
                      verify=VERIFY,
                      raw_root_dir=DATA_PATH)
        print(f"Frame extraction complete. Frames saved to: {SAVE_FRAMES_ROOT}")

    def __str2bool(self, v: str) -> bool:
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="YouCook2 Video Processing Pipeline with Enhanced Decorators"
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
            default="parta",
            help="Download part (default: parta)"
        )

        parser.add_argument(
            "--max_videos",
            type=int,
            default=3,
            help="Max number of videos to process (default: 3)"
        )

        # New arguments for decorator configuration
        parser.add_argument(
            "--disable_reverse",
            action="store_true",
            help="Disable reverse video generation"
        )

        parser.add_argument(
            "--disable_speed",
            action="store_true", 
            help="Disable speed change video generation"
        )

        parser.add_argument(
            "--disable_temporal_order",
            action="store_true",
            help="Disable temporal order video generation"
        )
            
        return parser.parse_args()

    def _print_pipeline_summary(self, 
                               skip_download_data,
                               skip_create_chunks, 
                               skip_apply_decorators,
                               skip_extract_frames,
                               max_videos):
        print("\n" + "="*60)
        print("VIDEO PROCESSING PIPELINE SUMMARY")
        print("="*60)
        print(f"Data Path: {DATA_PATH}")
        print(f"Max Videos: {max_videos}")
        print(f"Max Workers: {MAX_WORKERS}")
        print(f"")
        print("Pipeline Steps:")
        print(f"  1. Download Data: {'SKIP' if skip_download_data else 'RUN'}")
        print(f"  2. Create Chunks: {'SKIP' if skip_create_chunks else 'RUN'}")
        print(f"  3. Apply Decorators: {'SKIP' if skip_apply_decorators else 'RUN'}")
        print(f"  4. Extract Frames: {'SKIP' if skip_extract_frames else 'RUN'}")
        
        if not skip_apply_decorators:
            print(f"\nDecorator Configuration:")
            print(f"  - Reverse Videos: {ENABLE_REVERSE}")
            print(f"  - Speed Changes: {ENABLE_SPEED_CHANGE}")
            print(f"  - Temporal Order: {ENABLE_TEMPORAL_ORDER}")
        print("="*60 + "\n")

    def main(self, 
             download_part="parta",
             skip_download_data=False,
             skip_create_chunks=False,
             skip_apply_decorators=False,
             skip_extract_frames=False, 
             max_videos=None,
             disable_reverse=False,
             disable_speed=False,
             disable_temporal_order=False):
        
        # Update global configuration based on arguments
        global ENABLE_REVERSE, ENABLE_SPEED_CHANGE, ENABLE_TEMPORAL_ORDER
        ENABLE_REVERSE = not disable_reverse
        ENABLE_SPEED_CHANGE = not disable_speed
        ENABLE_TEMPORAL_ORDER = not disable_temporal_order
        
        # Print pipeline summary
        self._print_pipeline_summary(skip_download_data, skip_create_chunks, 
                                   skip_apply_decorators, skip_extract_frames, max_videos)
        
        if not skip_download_data:
            # STEP 1: download the data into disk
            print("STEP 1: Downloading data...")
            DownloadData(part=download_part, save_folder=DATA_PATH).run()

        # STEP 2: Download the labels and annotations
        print("STEP 2: Extracting labels and annotations...")
        self.extract_labels_annotations()

        # STEP 3: analyze the annotations
        print("STEP 3: Analyzing annotations...")
        self.analyze_annotations()

        # STEP 4: Create chunks from videos
        if not skip_create_chunks:
            print("STEP 4: Creating video chunks...")
            self._create_chunks_driver(max_videos=max_videos)

        # STEP 5: Apply Decorators (with enhanced diversity)
        if not skip_apply_decorators:
            print("STEP 5: Applying decorators...")
            self._apply_decorators()

        # STEP 6: Save as frames
        if not skip_extract_frames:
            print("STEP 6: Extracting frames...")
            self._extract_frames()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

if __name__ == "__main__":
    C = CreateData()
    args = C.parse_args()
    C.main(download_part=args.download_part,
           skip_download_data=args.skip_download_data,
           skip_create_chunks=args.skip_create_chunks,
           skip_apply_decorators=args.skip_apply_decorators,
           skip_extract_frames=args.skip_extract_frames,
           max_videos=args.max_videos,
           disable_reverse=args.disable_reverse,
           disable_speed=args.disable_speed,
           disable_temporal_order=args.disable_temporal_order)