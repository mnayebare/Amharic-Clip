from datasets import load_dataset
import pandas as pd
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


DATASET_NAME = "laion/relaion2B-en-research-safe"

MAX_SCAN = 100000
MAX_SAMPLES = 1000

OUTPUT_CSV = "laion_cultural_food_events_filtered.csv"
OUTPUT_JSONL = "laion_cultural_food_events_filtered.jsonl"

CHECK_URLS = True
URL_TIMEOUT = 8
MAX_URL_WORKERS = 16


CATEGORY_KEYWORDS = {
    "food_cooking": [
       "food"
    ],
      "animals": [
       "dog",
       "cat"
    ],


    "clothing": [
        "traditional ceremony",
        "traditional dance",
        "traditional festival",
        "traditional wedding",
    ],
}


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def assign_categories(caption):
    caption_norm = normalize_text(caption)

    categories = []
    matched_keywords = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = []

        for kw in keywords:
            kw_norm = normalize_text(kw)

            if kw_norm in caption_norm:
                matches.append(kw)

        if matches:
            categories.append(category)
            matched_keywords[category] = matches

    return categories, matched_keywords


def is_reasonable_caption(caption):
    words = str(caption).split()

    if len(words) < 2:
        return False

    if len(words) > 40:
        return False

    return True


def url_exists(url):
    """
    Checks whether the image URL is likely alive.

    HEAD is faster, but some servers reject HEAD.
    So we try HEAD first, then fallback to GET with stream=True.
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.head(
            url,
            timeout=URL_TIMEOUT,
            allow_redirects=True,
            headers=headers
        )

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()

            if "image" in content_type:
                return True, response.status_code, content_type

        # Some image servers reject HEAD but allow GET
        response = requests.get(
            url,
            timeout=URL_TIMEOUT,
            allow_redirects=True,
            stream=True,
            headers=headers
        )

        content_type = response.headers.get("Content-Type", "").lower()

        if response.status_code == 200 and "image" in content_type:
            return True, response.status_code, content_type

        return False, response.status_code, content_type

    except Exception as e:
        return False, None, str(e)


def check_urls_parallel(records):
    valid_records = []

    print("\nChecking URL existence...")

    with ThreadPoolExecutor(max_workers=MAX_URL_WORKERS) as executor:
        future_to_record = {
            executor.submit(url_exists, r["url"]): r
            for r in records
        }

        for i, future in enumerate(as_completed(future_to_record)):
            record = future_to_record[future]

            is_alive, status_code, content_type = future.result()

            if i % 100 == 0:
                print(f"Checked {i}/{len(records)} URLs, valid so far: {len(valid_records)}")

            record["url_alive"] = is_alive
            record["url_status_code"] = status_code
            record["url_content_type"] = content_type

            if is_alive:
                valid_records.append(record)

    return valid_records


print("Loading LAION stream...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

candidate_records = []

print("Scanning dataset...")

for i, sample in enumerate(ds):

    if i % 1000 == 0:
        print(f"Scanned {i}, candidates found {len(candidate_records)}")

    if i >= MAX_SCAN:
        print("Reached scan limit.")
        break

    caption = sample.get("caption", "")
    url = sample.get("url", "")

    if not caption or not url:
        continue

    if not is_reasonable_caption(caption):
        continue

    categories, matched_keywords = assign_categories(caption)

    if not categories:
        continue

    record = {
        "url": url,
        "caption": caption,
        "categories": categories,
        "matched_keywords": matched_keywords,
        "similarity": sample.get("similarity", None),
        "width": sample.get("width", None),
        "height": sample.get("height", None),
        "punsafe": sample.get("punsafe", None),
        "pwatermark": sample.get("pwatermark", None),
        "key": sample.get("key", None),
    }

    candidate_records.append(record)

    if len(candidate_records) >= MAX_SAMPLES:
        print("Reached candidate sample limit.")
        break


print("\nInitial candidates:", len(candidate_records))

if CHECK_URLS:
    final_records = check_urls_parallel(candidate_records)
else:
    final_records = candidate_records


df = pd.DataFrame(final_records)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in final_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


print("\nDONE")
print(f"Candidates before URL check: {len(candidate_records)}")
print(f"Valid image URLs after check: {len(final_records)}")
print(f"CSV: {OUTPUT_CSV}")
print(f"JSONL: {OUTPUT_JSONL}")

if len(final_records) > 0:
    print("\nSample rows:")
    print(df.head())