"""
download_pacs_hf.py
-------------------
Downloads the PACS dataset from Hugging Face (flwrlabs/pacs) and
organises it into the folder structure that DomainBed expects:

    ./data/PACS/
        art_painting/
            dog/  elephant/  giraffe/  guitar/  horse/  house/  person/
        cartoon/   ...
        photo/     ...
        sketch/    ...

Usage (from DomainBed directory, with venv activated):
    pip install datasets pillow
    python download_pacs_hf.py
"""

import os
from pathlib import Path

DATA_DIR = Path("./data/PACS")

print("Installing required packages...")
os.system("pip install -q datasets pillow")

from datasets import load_dataset
from PIL import Image

DOMAIN_MAP = {
    0: "art_painting",
    1: "cartoon",
    2: "photo",
    3: "sketch",
}

LABEL_MAP = {
    0: "dog",
    1: "elephant",
    2: "giraffe",
    3: "guitar",
    4: "horse",
    5: "house",
    6: "person",
}

def save_split(split_name, domain_map, label_map, data_dir):
    print(f"\nDownloading split: {split_name} ...")
    ds = load_dataset("flwrlabs/pacs", split=split_name, trust_remote_code=True)

    for i, sample in enumerate(ds):
        domain_name = domain_map.get(sample["domain"], f"domain_{sample['domain']}")
        label_name  = label_map.get(sample["label"],  f"class_{sample['label']}")

        out_dir = data_dir / domain_name / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        img: Image.Image = sample["image"]
        # Use a zero-padded index as filename to avoid collisions across splits
        img_path = out_dir / f"{split_name}_{i:06d}.jpg"
        if not img_path.exists():
            img.save(img_path, "JPEG", quality=95)

        if i % 500 == 0:
            print(f"  [{split_name}] saved {i} images...", flush=True)

    print(f"  [{split_name}] done — {i+1} images saved.")


if __name__ == "__main__":
    print(f"Saving dataset to: {DATA_DIR.resolve()}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        save_split(split, DOMAIN_MAP, LABEL_MAP, DATA_DIR)

    print("\n=== Done ===")
    print("Directory structure:")
    for domain in DOMAIN_MAP.values():
        d = DATA_DIR / domain
        if d.exists():
            classes = [c.name for c in sorted(d.iterdir()) if c.is_dir()]
            counts  = [len(list((d/c).glob("*.jpg"))) for c in classes]
            print(f"  {domain}/  ({sum(counts)} images)")
            for c, n in zip(classes, counts):
                print(f"    {c}/  ({n} images)")

    print(f"\nPACS is ready at: {DATA_DIR.resolve()}")
    print("You can now run:  python run_fair_comparison.py")
