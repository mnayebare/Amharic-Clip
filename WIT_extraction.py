"""
WIT extraction script with broad African cultural categorization.

Categories:
1. geography_identity
2. clothing_appearance
3. food_cooking
4. daily_life_practices

Requirements:
    pip install pandas requests pillow tqdm

Place this script in the same folder as:
    wit_v1.train.all-00009-of-00010.tsv.gz
"""

import hashlib
import json
import re
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

WIT_FILE = "wit_v1.train.all-00009-of-00010.tsv.gz"

OUTPUT_DIR = "wit_african_culture_output"

MAX_ROWS = 200000
MAX_IMAGES = 5000
CHUNK_SIZE = 50000


# =========================================================
# CATEGORY KEYWORDS
# =========================================================

GEOGRAPHY_IDENTITY = [
    "africa", "african",
    "ethiopia", "ethiopian",
    "ghana", "ghanaian",
    "nigeria", "nigerian",
    "kenya", "kenyan",
    "uganda", "ugandan",
    "tanzania", "tanzanian",
    "senegal", "senegalese",
    "somalia", "somali",
    "eritrea", "eritrean",
    "sudan", "sudanese",
    "maasai", "yoruba", "igbo", "zulu", "habesha",

    # Amharic / Ethiopic
    "ኢትዮጵያ", "አፍሪካ", "ሀበሻ", "ሐበሻ"
]

CLOTHING_APPEARANCE = [
    "traditional clothing",
    "traditional dress",
    "traditional clothes",
    "robe",
    "head wrap",
    "headscarf",
    "textile",
    "fabric",
    "woven",
    "pattern",
    "beads",
    "jewelry",
    "garment",
    "fashion",
    "dress",
    "cloth",
    "kente",
    "ankara",
    "dashiki",
    "habesha kemis",
    "beaded necklace"
]

FOOD_COOKING = [
    "food",
    "meal",
    "bread",
    "stew",
    "coffee",
    "cooking",
    "kitchen",
    "restaurant",
    "spices",
    "dish",
    "market food",
    "injera",
    "doro wat",
    "doro wot",
    "fufu",
    "jollof",
    "ugali",
    "shiro",
    "tibs",

    # Amharic
    "እንጀራ", "ኤንጀራ", "ዶሮ ወጥ", "ሽሮ", "ጥብስ", "ቡና"
]

DAILY_LIFE_PRACTICES = [
    "market",
    "street",
    "festival",
    "dance",
    "music",
    "ceremony",
    "wedding",
    "craft",
    "farming",
    "village",
    "family",
    "community",
    "drumming",
    "ritual",
    "celebration",
    "worker",
    "farmer",
    "women carrying",
    "coffee ceremony"
]


CATEGORY_KEYWORDS = {
    "geography_identity": GEOGRAPHY_IDENTITY,
    "clothing_appearance": CLOTHING_APPEARANCE,
    "food_cooking": FOOD_COOKING,
    "daily_life_practices": DAILY_LIFE_PRACTICES,
}


TEXT_COLUMNS = [
    "caption_reference_description",
    "caption_alt_text_description",
    "contextual_text",
    "page_title",
    "section_title",
]


# =========================================================
# HELPERS
# =========================================================

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(text):
    text = safe_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text_blob(row):
    parts = []
    for col in TEXT_COLUMNS:
        if col in row:
            parts.append(safe_text(row[col]))
    return normalize_text(" ".join(parts))


def choose_caption(row):
    """
    Choose the most caption-like field.
    Priority:
    1. caption_reference_description
    2. caption_alt_text_description
    3. contextual_text
    4. page_title
    """
    for col in TEXT_COLUMNS:
        if col in row:
            txt = safe_text(row[col])
            if len(txt.split()) >= 2:
                return txt
    return ""


def assign_categories(text_blob):
    """
    Returns:
        categories: list of matched category names
        matched_keywords: dict of category -> matched terms
    """
    categories = []
    matched_keywords = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = []

        for kw in keywords:
            kw_norm = normalize_text(kw)

            if kw_norm in text_blob:
                matches.append(kw)

        if matches:
            categories.append(category)
            matched_keywords[category] = matches

    return categories, matched_keywords


def is_simple_visual_caption(caption):
    """
    Keeps captions that are not too short or too long.
    Adjust as needed.
    """
    words = caption.split()
    return 2 <= len(words) <= 30


def download_image(url, save_path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(
            url,
            timeout=10,
            headers=headers
        )

        if response.status_code != 200:
            return False

        if len(response.content) < 1024:
            return False

        image = Image.open(BytesIO(response.content))
        image.verify()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception:
        return False


# =========================================================
# MAIN
# =========================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    image_dir = out_dir / "images"

    out_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)

    records = []
    seen_urls = set()
    total_rows = 0

    category_counts = {
        "geography_identity": 0,
        "clothing_appearance": 0,
        "food_cooking": 0,
        "daily_life_practices": 0,
    }

    print(f"\nLoading WIT file: {WIT_FILE}")

    chunks = pd.read_csv(
        WIT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="skip"
    )

    for chunk in chunks:
        total_rows += len(chunk)

        if total_rows > MAX_ROWS:
            chunk = chunk.head(
                MAX_ROWS - (total_rows - len(chunk))
            )

        print(f"\nProcessing rows up to: {min(total_rows, MAX_ROWS)}")

        for _, row in tqdm(chunk.iterrows(), total=len(chunk)):

            if len(records) >= MAX_IMAGES:
                break

            url = safe_text(row.get("image_url", ""))

            if not url:
                continue

            if url in seen_urls:
                continue

            caption = choose_caption(row)

            if not caption:
                continue

            if not is_simple_visual_caption(caption):
                continue

            text_blob = build_text_blob(row)

            categories, matched_keywords = assign_categories(text_blob)

            # Keep only rows that match at least one of the four domains
            if not categories:
                continue

            url_hash = hashlib.md5(
                url.encode("utf-8")
            ).hexdigest()

            image_path = image_dir / f"{url_hash}.jpg"

            success = download_image(url, image_path)

            if not success:
                continue

            seen_urls.add(url)

            for cat in categories:
                category_counts[cat] += 1

            record = {
                "id": url_hash,
                "image_path": str(image_path),
                "image_url": url,
                "caption": caption,
                "language": safe_text(row.get("language", "")).lower(),
                "categories": categories,
                "matched_keywords": matched_keywords,
                "page_title": safe_text(row.get("page_title", "")),
                "contextual_text": safe_text(row.get("contextual_text", "")),
                "caption_reference_description": safe_text(
                    row.get("caption_reference_description", "")
                ),
                "caption_alt_text_description": safe_text(
                    row.get("caption_alt_text_description", "")
                ),
            }

            records.append(record)

        if len(records) >= MAX_IMAGES:
            break

        if total_rows >= MAX_ROWS:
            break

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    df = pd.DataFrame(records)

    csv_path = out_dir / "wit_african_culture_metadata.csv"
    jsonl_path = out_dir / "wit_african_culture_clip_ready.jsonl"
    summary_path = out_dir / "category_summary.json"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            clip_record = {
                "image": r["image_path"],
                "caption": r["caption"],
                "language": r["language"],
                "categories": r["categories"],
                "matched_keywords": r["matched_keywords"],
                "source": "WIT"
            }
            f.write(json.dumps(clip_record, ensure_ascii=False) + "\n")

    summary = {
        "rows_scanned": min(total_rows, MAX_ROWS),
        "images_downloaded": len(records),
        "category_counts": category_counts,
        "output_csv": str(csv_path),
        "output_jsonl": str(jsonl_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"Rows scanned: {min(total_rows, MAX_ROWS)}")
    print(f"Images downloaded: {len(records)}")
    print("\nCategory counts:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

    print(f"\nCSV saved: {csv_path}")
    print(f"JSONL saved: {jsonl_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()