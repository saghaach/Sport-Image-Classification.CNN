"""
Module d'extraction multi-catégories pour Wikimedia Commons.
Utilise l'API de recherche plein texte (generator=search) pour garantir
l'obtention de photographies de haute qualité pour toutes les disciplines.
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

# Requêtes de recherche ciblées
SEARCH_QUERIES = {
    "football": "filetype:bitmap association football match action",
    "basketball": "filetype:bitmap basketball game player action",
    "tennis": "filetype:bitmap tennis match player serve"
}

TARGET_PER_CLASS = 40
RANDOM_SEED = 42

HEADERS = {
    "User-Agent": "AcademicSportClassifierBot/1.0 (academic university project; contact: student@univ.edu)"
}

def search_wikimedia_images(query, max_images=60):
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(max_images),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata"
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    
    entries = []
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
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
        print(f"Erreur API pour la requête '{query}': {e}")
        
    return entries

def collect_missing_categories():
    print("=== Collecte des catégories Basketball et Tennis ===")
    random.seed(RANDOM_SEED)
    
    sources_meta_file = DATASET_DIR / "sources_metadata.json"
    metadata_log = {}
    if sources_meta_file.exists():
        try:
            with open(sources_meta_file, "r", encoding="utf-8") as f:
                metadata_log = json.load(f)
        except Exception:
            metadata_log = {}

    for cat_name in ["basketball", "tennis"]:
        query = SEARCH_QUERIES[cat_name]
        cat_raw_dir = RAW_DIR / cat_name
        cat_raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier si des images existent déjà
        existing = list(cat_raw_dir.glob("*.jpg"))
        if len(existing) >= TARGET_PER_CLASS:
            print(f"✓ {cat_name} possède déjà {len(existing)} images.")
            continue
            
        print(f"\nRecherche d'images pour '{cat_name.upper()}' (requête : {query})...")
        candidates = search_wikimedia_images(query, max_images=TARGET_PER_CLASS + 25)
        print(f"  -> {len(candidates)} photographies candidates identifiées.")
        
        downloaded = len(existing)
        cat_meta = []
        
        for entry in candidates:
            if downloaded >= TARGET_PER_CLASS:
                break
                
            img_url = entry["url"]
            filename = f"{cat_name}_{downloaded+1:03d}.jpg"
            dest_path = cat_raw_dir / filename
            
            try:
                req = urllib.request.Request(img_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    
                with open(dest_path, "wb") as f:
                    f.write(data)
                    
                # Vérification d'intégrité
                with Image.open(dest_path) as img:
                    img.verify()
                    
                # Normalisation en RGB
                with Image.open(dest_path) as img:
                    rgb = img.convert("RGB")
                    rgb.thumbnail((600, 600), Image.Resampling.LANCZOS)
                    rgb.save(dest_path, "JPEG", quality=88)
                    
                downloaded += 1
                entry["local_filename"] = filename
                cat_meta.append(entry)
                print(f"  ✓ [{downloaded}/{TARGET_PER_CLASS}] {filename} validé ({entry['license']})")
                time.sleep(0.1)
                
            except Exception as e:
                if dest_path.exists():
                    dest_path.unlink()
                continue
                
        metadata_log[cat_name] = cat_meta
        print(f"✓ Total final pour {cat_name} : {downloaded} photographies valides.")
        
    with open(sources_meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_log, f, ensure_ascii=False, indent=2)

def organize_all_splits():
    print("\n=== Découpage étanche final (70% Train, 15% Val, 15% Test) ===")
    random.seed(RANDOM_SEED)
    
    splits = ["train", "validation", "test"]
    categories = ["football", "basketball", "tennis"]
    
    for s in splits:
        for c in categories:
            d = DATASET_DIR / s / c
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            
    recap = {}
    for cat_name in categories:
        cat_raw_dir = RAW_DIR / cat_name
        images = sorted(list(cat_raw_dir.glob("*.jpg")))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        train_set = images[:n_train]
        val_set = images[n_train:n_train + n_val]
        test_set = images[n_train + n_val:]
        
        for f in train_set:
            shutil.copy2(f, DATASET_DIR / "train" / cat_name / f.name)
        for f in val_set:
            shutil.copy2(f, DATASET_DIR / "validation" / cat_name / f.name)
        for f in test_set:
            shutil.copy2(f, DATASET_DIR / "test" / cat_name / f.name)
            
        recap[cat_name] = {
            "total": n_total,
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set)
        }
        print(f"[{cat_name.upper():<10}] Total: {n_total:2d} -> Train: {len(train_set):2d} (70%), Val: {len(val_set):2d} (15%), Test: {len(test_set):2d} (15%)")
        
    print("\n✓ Jeu de données complet équilibré et partitionné.")

if __name__ == "__main__":
    collect_missing_categories()
    organize_all_splits()
