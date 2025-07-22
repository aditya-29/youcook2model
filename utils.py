from pathlib import Path
from typing import Dict, List, Optional
import json

def get_caption_for_chunk(path, video_annotations):
    path = str(path)
    suffix = ""
    if "_tr_" in path:
        if "RCS" in path:
            if "x0_5" in path:
                suffix = " played in 0.5x speed"
            elif "x2" in path:
                suffix = " played in 2x speed"
            # ``x1`` adds no suffix
        elif "RR" in path:
            suffix = " played in reverse"
        # elif "RP" in path:
            

    subset = path.split("raw_annot_videos/")[1].split("/")[0]
    recipie = path.split(subset+"/")[1].split("/")[0]
    video_name_temp = path.split(subset+"/"+recipie+"/")[1]
    video_name = "__".join(video_name_temp.split("__")[:-1])
    # video_name = path.split(subset+"/"+recipie+"/")[1].split("__")[0]
    id = path.split("__")[-1].split(".mp4")[0].split("_")[0]

    # print(subset, recipie, video_name, id)

    for iter in range(len(video_annotations[video_name])):
        if video_annotations[video_name][iter]["id"] == int(id):
            return video_annotations[video_name][iter]["sentence"] + suffix

    return None

# class Captions:
#     def __init__(self, annotations_json_path):
#         self.annotations_json_path = annotations_json_path
#         self.VIDEO_ANNOTATIONS = None
#         self.load_annotations()

#     def load_annotations(self,):
#         if self.VIDEO_ANNOTATIONS is None:
#             with open(self.annotations_json_path, "r") as f:
#                 self.VIDEO_ANNOTATIONS = json.load(f)
        

#     def get_caption_for_chunk(path: str) -> Optional[str]:
#         """Mimics the original caption‑lookup logic but works in every process."""

#         suffix = ""
#         if "_tr_" in path:
#             if "RCS" in path:
#                 if "x0_5" in path:
#                     suffix = " played in 0.5x speed"
#                 elif "x2" in path:
#                     suffix = " played in 2x speed"
#                 # ``x1`` adds no suffix
#             elif "RR" in path:
#                 suffix = " played in reverse"

#         base_path = path.split("_tr_")[0] + ".mp4"
#         subset = base_path.split("raw_annot_videos/")[1].split("/")[0]
#         recipe = base_path.split(f"{subset}/")[1].split("/")[0]
#         video_name = base_path.split(f"{subset}/{recipe}/")[1].split("__")[0]
#         chunk_id = int(base_path.split("__")[1].split(".mp4")[0])

#         for meta in annotations.get(video_name, []):
#             if meta["id"] == chunk_id:
#                 return meta["sentence"] + suffix
#         return None
