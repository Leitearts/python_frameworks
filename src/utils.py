import re
from collections import Counter
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    keep = [c for c in ["cord_uid","title","abstract","publish_time","journal","source_x","url"] if c in df.columns]
    return df[keep].copy() if keep else df

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["year"] = df["publish_time"].dt.year
    else:
        df["year"] = pd.NA
    if "title" in df.columns:
        df = df.dropna(subset=["title"])
        df["title_word_count"] = df["title"].fillna("").str.split().str.len()
    return df

def common_title_words(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    stop = set("a an and are as at be by for from has have in is it its of on or that the to was were with covid coronavirus study".split())
    counts = Counter()
    for t in df.get("title", pd.Series(dtype=str)).dropna().tolist():
        for w in re.findall(r"[a-zA-Z]+", t.lower()):
            if len(w) > 2 and w not in stop:
                counts.update([w])
    return pd.DataFrame(counts.most_common(top_n), columns=["word","count"])

def ensure_dir(path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
