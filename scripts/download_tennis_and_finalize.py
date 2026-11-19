"""
Téléchargement finalisé pour Tennis et Basketball et découpage 70/15/15.
"""

import os
import sys
import json
import time
import shutil
import random
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"

RANDOM_SEED = 42
TARGET_PER_CLASS = 35

HEADERS = {
    "User-Agent": "AcademicSportClassifierBot/1.0 (academic university project; contact: student@univ.edu)"
}

def search_wikimedia(query, limit=50):
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata"
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    entries = []
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                infos = pdata.get("imageinfo", [])
                if infos:
                    info = infos[0]
                    mime = info.get("mime", "")
                    url_img = info.get("url", "")
                    if mime in ["image/jpeg", "image/png"] and url_img:
                        meta = info.get("extmetadata", {})
                        lic = meta.get("LicenseShortName", {}).get("value", "CC-BY / Public Domain")
                        entries.append({
                            "title": pdata.get("title", ""),
                            "url": url_img,
                            "license": lic
                        })
    except Exception as e:
        print(f"Erreur API pour {query}: {e}")
    return entries

def download_category(cat_name, queries, target=TARGET_PER_CLASS):
    cat_dir = RAW_DIR / cat_name
    cat_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cat_dir.glob("*.jpg"))
    downloaded = len(existing)
    
    if downloaded >= target:
        print(f"✓ {cat_name} a déjà {downloaded} images.")
        return downloaded
        
    print(f"\nTéléchargement pour '{cat_name.upper()}' (actuellement: {downloaded}/{target})...")
    
    seen_urls = set()
    for q in queries:
        if downloaded >= target:
            break
        print(f"  Recherche avec : '{q}'...")
        candidates = search_wikimedia(q, limit=50)
        for entry in candidates:
            if downloaded >= target:
                break
            url_img = entry["url"]
            if url_img in seen_urls:
                continue
            seen_urls.add(url_img)
            
            filename = f"{cat_name}_{downloaded+1:03d}.jpg"
            dest_path = cat_dir / filename
            try:
                req = urllib.request.Request(url_img, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                with open(dest_path, "wb") as f:
                    f.write(data)
                with Image.open(dest_path) as img:
                    img.verify()
                with Image.open(dest_path) as img:
                    rgb = img.convert("RGB")
                    rgb.thumbnail((600, 600), Image.Resampling.LANCZOS)
                    rgb.save(dest_path, "JPEG", quality=88)
                downloaded += 1
                print(f"  ✓ [{downloaded}/{target}] {filename} ({entry['license']})")
                time.sleep(0.1)
            except Exception:
                if dest_path.exists():
                    dest_path.unlink()
                continue
    print(f"✓ {cat_name} finalisé avec {downloaded} photographies.")
    return downloaded

def partition_dataset():
    print("\n=== Partitionnement équilibré (70% Train, 15% Val, 15% Test) ===")
    random.seed(RANDOM_SEED)
    splits = ["train", "validation", "test"]
    categories = ["football", "basketball", "tennis"]
    
    for s in splits:
        for c in categories:
            d = DATASET_DIR / s / c
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            
    summary = {}
    for c in categories:
        imgs = sorted(list((RAW_DIR / c).glob("*.jpg")))
        random.shuffle(imgs)
        
        # Limiter à 35 par classe pour un équilibre parfait des 3 classes
        imgs = imgs[:35]
        n_total = len(imgs)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]
        
        for f in train_imgs:
            shutil.copy2(f, DATASET_DIR / "train" / c / f.name)
        for f in val_imgs:
            shutil.copy2(f, DATASET_DIR / "validation" / c / f.name)
        for f in test_imgs:
            shutil.copy2(f, DATASET_DIR / "test" / c / f.name)
            
        summary[c] = {
            "total": n_total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }
        print(f"[{c.upper():<10}] Total: {n_total:2d} -> Train: {len(train_imgs):2d} (70%), Val: {len(val_imgs):2d} (15%), Test: {len(test_imgs):2d} (15%)")
        
    print("\n✓ Jeu de données rigoureusement équilibré et partitionné !")
    return summary

if __name__ == "__main__":
    download_category("tennis", ["filetype:bitmap tennis player", "filetype:bitmap tennis match", "filetype:bitmap tennis court action"])
    download_category("basketball", ["filetype:bitmap basketball player", "filetype:bitmap basketball game"])
    partition_dataset()
