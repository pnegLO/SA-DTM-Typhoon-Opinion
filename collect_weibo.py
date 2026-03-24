import os
import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)


SEARCH_KEYWORDS = [
    "台风桦加沙", "桦加沙台风", "台风预警 桦加沙",
    "桦加沙 登陆", "桦加沙 暴雨", "桦加沙 救援",
    "#台风桦加沙#", "#桦加沙最新消息#", "#台风实时路径#",
]

WEIBO_SEARCH_API = "https://m.weibo.cn/api/container/getIndex"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                  "Mobile/15E148 MicroMessenger/8.0.38",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}

OUTPUT_PATH = "data/weibo_typhoon_raw.csv"

DATE_RANGE = (datetime(2025, 7, 15), datetime(2025, 7, 28))


def search_weibo(keyword, page=1, cookie=None):
    params = {
        "containerid": f"100103type=1&q={keyword}",
        "page_type": "searchall",
        "page": page,
    }
    headers = HEADERS.copy()
    if cookie:
        headers["Cookie"] = cookie

    try:
        resp = requests.get(WEIBO_SEARCH_API, params=params,
                            headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[WARN] Request failed: {e}")
    return None


def parse_card(card):
    mblog = card.get("mblog", {})
    if not mblog:
        return None

    text_raw = mblog.get("text", "")

    import re
    text_clean = re.sub(r"<[^>]+>", "", text_raw).strip()
    if len(text_clean) < 5:
        return None

    user = mblog.get("user", {})
    created_at = mblog.get("created_at", "")

    return {
        "post_id": mblog.get("id", ""),
        "user_id": user.get("id", ""),
        "username": user.get("screen_name", ""),
        "text": text_clean,
        "publish_time": created_at,
        "location": user.get("location", ""),
        "province": "",
        "likes": mblog.get("attitudes_count", 0),
        "reposts": mblog.get("reposts_count", 0),
        "comments": mblog.get("comments_count", 0),
        "is_original": 1 if mblog.get("retweeted_status") is None else 0,
    }


def collect(cookie=None, max_pages=50, delay=3.0):
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        print(f"[INFO] Data already exists: {OUTPUT_PATH} ({len(df)} posts)")
        print("[INFO] To re-collect, delete the file first.")
        return df

    all_posts = []

    for kw in SEARCH_KEYWORDS:
        print(f"\n[INFO] Searching: {kw}")
        for page in range(1, max_pages + 1):
            data = search_weibo(kw, page=page, cookie=cookie)
            if data is None:
                print(f"  Page {page}: request failed, skipping keyword")
                break

            cards = data.get("data", {}).get("cards", [])
            if not cards:
                print(f"  Page {page}: no more results")
                break

            count = 0
            for card in cards:
                if card.get("card_type") == 9:
                    post = parse_card(card)
                    if post:
                        all_posts.append(post)
                        count += 1

            print(f"  Page {page}: collected {count} posts (total: {len(all_posts)})")
            time.sleep(delay)

    if not all_posts:
        print("[WARN] No posts collected. Check cookie or network.")
        return None

    df = pd.DataFrame(all_posts)
    df = df.drop_duplicates(subset=["post_id"], keep="first")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved {len(df)} posts to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    cookie = os.environ.get("WEIBO_COOKIE", "")
    if not cookie:
        print("[INFO] No WEIBO_COOKIE set. Checking existing data...")
    collect(cookie=cookie if cookie else None)
