import requests
from pathlib import Path
import os
import json
from typing import List, Dict, Union

import sys
from pathlib import Path
# put project root on sys.path (…/repo_root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# tell Python what package this module belongs to (PEP 366)
__package__ = "youcook2model.video_comp"

from ..download_data import DownloadData as LegacyDownloadData

TRAIN_URL = "https://storage.googleapis.com/video_comp/youcook2_comp_train.json"
TEST_URL = "https://storage.googleapis.com/video_comp/youcook2_comp_val.json"


def _normalize_json(obj: Union[Dict, List]) -> List[Dict]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON structure")

def fetch_list_of_dicts(url: str, timeout: int = 60) -> List[Dict]:
    """
    Fetch URL that may be JSON array/object or JSONL (NDJSON).
    Returns a list[dict].
    """
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        # First try standard JSON
        try:
            return _normalize_json(r.json())
        except ValueError:
            # Fall back to JSONL/NDJSON
            items: List[Dict] = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                items.append(json.loads(line))
            return items

def save_list_of_dicts(objs: List[Dict], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(objs, ensure_ascii=False, indent=2), encoding="utf-8")




class DownloadData:
    def __init__(self, part="parta", save_folder="./data", file_name = "youcook2_video_comp.json"):
        self.save_folder = Path(save_folder)
        self.download_part = part
        self.file_name = Path(file_name)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        if part not in ["parta", "partb", "partc", "partd", "parte", "partf", "partg", "parth", "parti", "partj", "all"]:
            raise Exception("Invalid part")


    def download_raw_videos(self):
        LegacyDownloadData(part = self.download_part, save_folder = "/mnt/raw_videos")
            
    def run(self,):
        self.download_raw_videos()
        out_path = self.save_folder / self.file_name

        # ── Usage ─────────────────────────────────────────────
        train_items = fetch_list_of_dicts(TRAIN_URL, timeout=60)
        test_items  = fetch_list_of_dicts(TEST_URL,  timeout=60)
        
        all_items = train_items + test_items
        save_list_of_dicts(all_items, out_path)


        print("[INFO] Data downloaded successfully")