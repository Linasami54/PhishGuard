# downloads only the url and label columns from the PhreshPhish dataset (Hugging Face)
# we never open the links and we never download the html pages
# usage: python download_phreshphish.py

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

REPO = "phreshphish/phreshphish"
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"   # fixed version so the data is always the same
SHARDS_PER_SPLIT = 4
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data"


# lets pyarrow read parts of a remote file, so we only download the columns we need
class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?limit=1000", timeout=60).json()
    for split in ["train", "test"]:
        shards = [f for f in files if f["path"].startswith(f"data/{split}-")]
        chosen = [shards[int(i)] for i in np.linspace(0, len(shards) - 1, SHARDS_PER_SPLIT)]
        frames = []
        for f in chosen:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{f['path']}"
            table = pq.ParquetFile(RemoteFile(url, f["size"])).read(columns=["url", "label"])
            frames.append(table.to_pandas())
            print(split, f["path"], len(frames[-1]), "rows")
        df = pd.concat(frames, ignore_index=True)
        df["label"] = df["label"].map({"benign": 0, "phish": 1})   # 1 = phishing
        df = df.rename(columns={"url": "URL"}).dropna()
        df.to_csv(OUT_DIR / f"phreshphish_{split}_urls.csv", index=False)
        print("saved", OUT_DIR / f"phreshphish_{split}_urls.csv", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
