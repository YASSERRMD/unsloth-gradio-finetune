import os
import json
import hashlib
import pandas as pd
from typing import Tuple, List, Optional
from datasets import load_dataset


def load_hf_dataset(
    dataset_name: str, subset: Optional[str], split: str, n_preview: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    ds = load_dataset(dataset_name, subset, split=split)
    df = ds.to_pandas().head(n_preview)
    return df, list(df.columns)


def save_uploaded_file(file_obj, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(file_obj.name))
    with open(file_obj.name, "rb") as f:
        with open(dest, "wb") as out:
            out.write(f.read())
    return dest


def load_local_file(path: str, n_preview: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    if path.endswith(".csv"):
        df = pd.read_csv(path).head(n_preview)
    elif path.endswith(".jsonl"):
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= n_preview:
                    break
        df = pd.DataFrame(records)
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data[:n_preview])
        else:
            df = pd.DataFrame([data])
    return df, list(df.columns)
