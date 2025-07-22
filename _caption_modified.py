import json
from pathlib import Path
import os
from tqdm import tqdm
from main import CreateData
import argparse

OUTPUT_FILE = "./data/captions_all.json"

def get_caption_for_chunk(path, video_annotations):
    path = str(path)
    suffix = ""
    prefix = ""
    if "_tr_" in path:
        if "_RCS" in path:
            if "x0_5" in path:
                suffix = " played in 0.5x speed"
            elif "x2" in path:
                suffix = " played in 2x speed"
            # ``x1`` adds no suffix
        elif "_RR" in path:
            suffix = " played in reverse"
        elif "_RP" in path:
            print("asd")
            prefix = "part of "

    subset = path.split("raw_annot_videos/")[1].split("/")[0]
    recipie = path.split(subset+"/")[1].split("/")[0]
    video_name_temp = path.split(subset+"/"+recipie+"/")[1]
    video_name = "__".join(video_name_temp.split("__")[:-1])
    # video_name = path.split(subset+"/"+recipie+"/")[1].split("__")[0]
    id = path.split("__")[-1].split(".mp4")[0].split("_")[0]

    # print(subset, recipie, video_name, id)

    for iter in range(len(video_annotations[video_name])):
        if video_annotations[video_name][iter]["id"] == int(id):
            return prefix + video_annotations[video_name][iter]["sentence"] + suffix

    return None

def validator(annot_videos_path, caption_path):
    if not os.path.exists(caption_path):
        raise Exception(f"{caption_path} does not exists")
    
    if not os.path.exists(annot_videos_path):
        raise Exception(f"{annot_videos_path} does not exists")
    
    if ".json" not in caption_path:
        raise Exception("Please provide the caption.json file")


def load_video_captions(path: str | Path) -> dict[str, str]:
    """
    Read a “JSON‑Lines” file where each line looks like
       {"video": "...", "caption": "..."}
    and return a dict {video_path: caption}.
    """
    captions = {}
    with Path(path).expanduser().open(encoding="utf‑8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue                      # skip blank lines
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as err:
                # If you want to know which line failed:
                print(f"⚠️  Line {lineno}: {err.msg} — skipped")
                continue

            video = str(Path("annot_videos") / Path(obj.get("video")))
            caption = obj.get("caption")
            if video is not None and caption is not None:
                if "_RP" in video:
                    caption = "part of " + caption
                captions[video] = caption     # later duplicates overwrite earlier ones
            else:
                print(f"⚠️  Line {lineno}: missing 'video' or 'caption' key — skipped")

    return captions

def clean_current_caption_mp(caption_mp):
    output = {}
    for key, value in caption_mp.items():
        if "raw_annot_videos" in key:
            key = "raw_annot_videos/" + key.split("raw_annot_videos/")[-1]
        output[key] = value

    return output

def main(annot_videos_path, caption_path):
    validator(annot_videos_path=annot_videos_path, caption_path=caption_path)

    current_caption_mp = load_video_captions(caption_path)
    current_caption_mp = clean_current_caption_mp(current_caption_mp)
    output_caption_mp = {}

    C = CreateData()
    C.extract_labels_annotations()
    C.annalyze_annotations()
    video_annotations = C.video_annotations

    failures = []
    total_videos = 0

    for subset in tqdm(os.listdir(annot_videos_path)):
        subset_path = os.path.join(annot_videos_path, subset)

        for recipie in os.listdir(subset_path):
            recipie_path = os.path.join(subset_path, recipie)

            for video in os.listdir(recipie_path):
                total_videos += 1
                video_path = os.path.join(recipie_path, video)
                video_path = video_path.split("annot_videos/")[-1]

                __check_video_path = "annot_videos/" + video_path
                if __check_video_path in current_caption_mp or "_RP" in video_path:
                    continue

                try: 
                    video_caption = get_caption_for_chunk("raw_annot_videos/" + video_path, video_annotations)
                except:
                    failures.append("annot_videos/" + video_path)
                    
                if video_caption == None:
                    raise Exception(f"Video Caption not found : {video_path}")
                
                current_caption_mp["annot_videos/" + video_path] = video_caption
                
    print("total videos found : ", total_videos)
    print("total failures : ", len(failures))
    print("total captions available : ", len(current_caption_mp))

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(current_caption_mp, f)

def parser():
    parser = argparse.ArgumentParser(description = "1 time caption evolver")
    parser.add_argument("--annot_videos_path", required=True, help="Root directory where the annot_videos are saved")
    parser.add_argument("--caption_path", required=True, help = "Path where the caption.json files are saved")
    args = parser.parse_args()

    main(annot_videos_path = args.annot_videos_path,
        caption_path = args.caption_path)


if __name__ == "__main__":
    parser()