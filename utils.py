from pathlib import Path
from typing import Dict, List, Optional
import json

def get_caption_for_chunk(path, video_annotations):
    path = str(path)
    suffix = ""

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

