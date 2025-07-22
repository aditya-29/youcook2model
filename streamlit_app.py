#!/usr/bin/env python
# app.py – Streamlit UI for YouCook‑style decorator videos (enhanced ‑ merge explorer now in its own row)
#
# Usage:   streamlit run app.py
#
# Prereqs: pip install streamlit opencv-python  (OpenCV only if you want local playback too)

import json
import re
from pathlib import Path
import streamlit as st

# ────────────────────────────────
# 1.  Project‑specific constants
# ────────────────────────────────
RAW_ROOT = Path("./data/annot_videos")          # training / validation / testing
CAPTION_JSON = Path("./data/captions_all.json") # same format you used before
TRANS_LABELS = {
    "ORIG": "Original",
    "RR": "Random Reverse",
    "RCS_x0_5": "Speed 0.5×",
    "RCS_x1": "Speed 1×",
    "RCS_x2": "Speed 2×",
    "RP": "Random Part",
}

# ────────────────────────────────
# 2.  Helpers
# ────────────────────────────────
@st.cache_data(show_spinner=False)
def load_captions() -> dict[str, str]:
    if CAPTION_JSON.exists():
        with open(CAPTION_JSON, "r") as f:
            return json.load(f)
    return {}

CAPTIONS = load_captions()

def __modify_name(path: str) -> str:
    """Handle backward‑compatibility rename for merged clips."""
    video_file = Path(path).name
    video_name = video_file.split("__")[0]
    video_chunk_1 = video_file.split(video_name)
    video_chunk_1 = video_name + video_chunk_1[1][:-1]
    video_chunk_2 = video_file.split(video_chunk_1)[1][1:].split("_tr")[0]

    new_video_file = (
        video_chunk_1 + "_mrg_" + video_chunk_2 + video_file.split(video_chunk_2)[-1]
    )
    return str(Path(path).parent / new_video_file)

def get_caption(path: str, caption_mp: dict[str, str] = CAPTIONS) -> str | None:
    """Return caption for a transformed or raw clip path (fallback‑aware)."""
    orig_path = path
    if "raw_annot_videos" in path:
        path = "raw_annot_videos/" + path.split("raw_annot_videos/")[-1]
    elif "annot_videos/" in path:
        path = "annot_videos/" + path.split("annot_videos/")[-1]

    if path not in caption_mp:
        path = __modify_name(path)
    return caption_mp.get(path)

@st.cache_data(show_spinner=False)
def index_videos() -> dict[str, dict[str, Path]]:
    """
    Return {base_id: {transformation_id: Path, ...}, ...}
    where base_id is '<uniq_id>__<s.no>' (or merged pattern).
    """
    pat = re.compile(r"(.+?)_tr_([A-Za-z0-9x_]+)\.mp4$")  # matches <base>_tr_<trans>.mp4
    table: dict[str, dict[str, Path]] = {}

    for mp4 in RAW_ROOT.rglob("*.mp4"):
        name = mp4.name
        m = pat.match(name)
        if m:
            base_id, trans_id = m.groups()
            table.setdefault(base_id, {})[trans_id] = mp4
        else:
            # treat files without '_tr_' as originals
            table.setdefault(mp4.stem, {})["ORIG"] = mp4

    # We only keep fully‑decorated simple clips OR any merged clip
    table_simpl: dict[str, dict[str, Path]] = {}
    for key, value in table.items():
        if "_mrg_" in key:
            table_simpl[key] = value
        else:
            if len(value.keys()) >= 4:  # ORIG + 3 decorators
                table_simpl[key] = value
    return table_simpl

CATALOG = index_videos()

# ────────────────────────────────
# 3.  Streamlit UI
# ────────────────────────────────
st.set_page_config(page_title="Decorator Video Explorer", layout="wide")
st.title("🎬 Decorator Video Explorer")

if not CATALOG:
    st.error("No *.mp4 files found under ‘annot_videos’.")
    st.stop()

base_ids = sorted(CATALOG.keys())
sel_base = st.selectbox("Choose a clip (base id)", base_ids, key="sel_base")

trans_available = list(CATALOG[sel_base].keys())
ordered = [t for t in TRANS_LABELS if t in trans_available] + [t for t in trans_available if t not in TRANS_LABELS]

sel_trans = st.radio(
    "Transformation",
    ordered,
    format_func=lambda t: TRANS_LABELS.get(t, t).replace("_", " "),
)

vid_path = CATALOG[sel_base][sel_trans]

# ────────────────────────────────
# Row 1 – Preview & Caption
# ────────────────────────────────
col_v, col_c = st.columns([3, 2])
with col_v:
    st.subheader("Preview")
    st.video(str(vid_path))

with col_c:
    st.subheader("Caption")
    cap = get_caption(str(vid_path))
    st.markdown(f"**{cap}**" if cap else "_Caption not found._")

# ────────────────────────────────
# Row 2 – Merge Explorer (if applicable)
# ────────────────────────────────
st.markdown("---")
st.subheader("🧩 Merge Explorer")

if "_mrg_" not in sel_base:
    st.info("Choose a clip whose name contains `_mrg_` to explore its component parts.")
else:
    # Helper: pick RCS_x1 if available, else ORIG, else first
    def _pick_clip(base_id: str) -> Path | None:
        d = CATALOG.get(base_id)
        if not d:
            return None
        return d.get("RCS_x1") or d.get("ORIG") or next(iter(d.values()))

    id1, id2 = sel_base.split("_mrg_", 1)
    vid1_path = _pick_clip(id1)
    vid2_path = _pick_clip(id2)
    merge_path = CATALOG[sel_base].get("RP") or _pick_clip(sel_base)

    cap1 = get_caption(str(vid1_path)) if vid1_path else None
    cap2 = get_caption(str(vid2_path)) if vid2_path else None
    cap_merge = get_caption(str(merge_path)) if merge_path else None

    # Display in one full‑width row (3 columns)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Video 1")
        if vid1_path:
            st.video(str(vid1_path))
            st.markdown(f"**{cap1}**" if cap1 else "_no caption_")
        else:
            st.text("clip not found")

    with col2:
        st.markdown("#### Video 2")
        if vid2_path:
            st.video(str(vid2_path))
            st.markdown(f"**{cap2}**" if cap2 else "_no caption_")
        else:
            st.text("clip not found")

    with col3:
        st.markdown("#### Merged (RP)")
        if merge_path:
            st.video(str(merge_path))
            st.markdown(f"**{cap_merge}**" if cap_merge else "_no caption_")
        else:
            st.text("clip not found")

# Footer – show path of the currently selected transformation
st.markdown("---")
relative_path = vid_path.relative_to(RAW_ROOT.parent)
st.write("***File:***", relative_path)
