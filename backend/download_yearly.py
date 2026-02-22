"""JRDB 年度パックを一括ダウンロード・展開するスクリプト"""
import os
import io
import zipfile
import requests

JRDB_BASE_URL = "https://jrdb.com/member/datazip"
JRDB_USER = os.environ.get("JRDB_USER", "16120005")
JRDB_PASS = os.environ.get("JRDB_PASS", "38057460")
DEST_DIR = os.path.join(os.path.dirname(__file__), "database", "temp")

# (JRDBディレクトリ, JRDBコード, ローカルコード)
# ローカルコードが異なる場合はリネームする
DATA_TYPES = [
    ("Bac", "BAC", "BAC"),
    ("Cha", "CHA", "CHA"),
    ("Cyb", "CYB", "CYB"),
    ("Cs",  "CSA", "CZA"),
    ("Hjc", "HJC", "HJC"),
    ("Jo",  "JOA", "JOA"),
    ("Kab", "KAB", "KAB"),
    ("Kka", "KKA", "KKA"),
    ("Kyi", "KYI", "KYI"),
    ("Ks",  "KSA", "KZA"),
    ("Ot",  "OT",  "OT"),
    ("Ou",  "OU",  "OU"),
    ("Ov",  "OV",  "OV"),
    ("Ow",  "OW",  "OW"),
    ("Oz",  "OZ",  "OZ"),
    ("Sed", "SED", "SED"),
    ("Skb", "SKB", "SKB"),
    ("Tyb", "TYB", "TYB"),
    ("Ukc", "UKC", "UKC"),
]

YEARS = [2021, 2022, 2023, 2024, 2025]


def download_and_extract(jrdb_dir, jrdb_code, local_code, year):
    zip_name = f"{jrdb_code}_{year}.zip"
    url = f"{JRDB_BASE_URL}/{jrdb_dir}/{zip_name}"

    print(f"  {url} ... ", end="", flush=True)
    try:
        resp = requests.get(url, auth=(JRDB_USER, JRDB_PASS), timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"FAILED ({e})")
        return 0

    count = 0
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue
            # リネーム: CSA170108.txt → CZA170108.txt
            dest_name = name
            if jrdb_code != local_code:
                dest_name = name.replace(jrdb_code, local_code, 1)
            dest_path = os.path.join(DEST_DIR, dest_name)
            data = zf.read(name)
            with open(dest_path, "wb") as f:
                f.write(data)
            count += 1

    print(f"OK ({count} files)")
    return count


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    total = 0

    for year in YEARS:
        print(f"\n=== {year}年 ===")
        for jrdb_dir, jrdb_code, local_code in DATA_TYPES:
            total += download_and_extract(jrdb_dir, jrdb_code, local_code, year)

    print(f"\n完了! 合計 {total} ファイルをダウンロードしました。")
    print(f"保存先: {DEST_DIR}")


if __name__ == "__main__":
    main()
