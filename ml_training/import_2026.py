"""2026年1-2月のJRDBデータを一括ダウンロード＆インポート"""
import os
import io
import sys
import zipfile
import requests
from datetime import date, timedelta

# Django setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jrdb.settings")
import django
django.setup()

from database.management.import_BaseCommand import read_and_write

AUTH = ("16120005", "38057460")
BASE_URL = "https://jrdb.com/member/datazip"
TEMP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "backend", "database", "temp")

# テーブル定義: (local_code, jrdb_dir, jrdb_code, model_import_name)
TABLES = [
    ("UKC", "Ukc", "UKC", "import_UKC"),
    ("CZA", "Cs", "CSA", "import_CZA"),
    ("KZA", "Ks", "KSA", "import_KZA"),
    ("KAB", "Kab", "KAB", "import_KAB"),
    ("BAC", "Bac", "BAC", "import_BAC"),
    ("KYI", "Kyi", "KYI", "import_KYI"),
    ("CYB", "Cyb", "CYB", "import_CYB"),
    ("CHA", "Cha", "CHA", "import_CHA"),
    ("KKA", "Kka", "KKA", "import_KKA"),
    ("JOA", "Jo", "JOA", "import_JOA"),
    ("TYB", "Tyb", "TYB", "import_TYB"),
    ("HJC", "Hjc", "HJC", "import_HJC"),
    ("SED", "Sed", "SED", "import_SED"),
]

BEGIN = date(2026, 1, 1)
END = date(2026, 2, 22)


def download_all():
    """全テーブル×全日付をダウンロード"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    total = 0
    d = BEGIN
    while d <= END:
        date_str = d.strftime("%y%m%d")
        for local_code, jrdb_dir, jrdb_code, _ in TABLES:
            dest = os.path.join(
                TEMP_DIR, f"{local_code}{date_str}.txt")
            if os.path.isfile(dest):
                continue
            url = (f"{BASE_URL}/{jrdb_dir}/2026/"
                   f"{jrdb_code}{date_str}.zip")
            try:
                r = requests.get(url, auth=AUTH, timeout=30)
                if r.status_code == 200 and len(r.content) > 100:
                    with zipfile.ZipFile(
                            io.BytesIO(r.content)) as zf:
                        for name in zf.namelist():
                            if name.lower().endswith(".txt"):
                                data = zf.read(name)
                                with open(dest, "wb") as f:
                                    f.write(data)
                                total += 1
                                break
            except Exception:
                pass
        d += timedelta(days=1)
    print(f"  ダウンロード完了: {total} files")


def import_all():
    """ダウンロード済みファイルをDjango経由でインポート"""
    from django.core import management

    d = BEGIN
    dates = []
    while d <= END:
        dates.append(d)
        d += timedelta(days=1)

    for local_code, _, _, cmd_name in TABLES:
        count = 0
        for dt in dates:
            date_str = dt.strftime("%y%m%d")
            fname = os.path.join(
                TEMP_DIR, f"{local_code}{date_str}.txt")
            if not os.path.isfile(fname):
                continue
            count += 1
        if count > 0:
            print(f"  {local_code}: {count} files → importing...")
            sys.stdout.flush()
            management.call_command(
                cmd_name,
                find_begin_date=BEGIN,
                find_end_date=END,
                no_download=True,
            )
        else:
            print(f"  {local_code}: skip (no files)")


if __name__ == "__main__":
    print("=== 2026年データ ダウンロード ===")
    download_all()
    # Django management commands use relative paths from backend/
    backend_dir = os.path.join(
        os.path.dirname(__file__), "..", "backend")
    os.chdir(backend_dir)
    print(f"  CWD → {os.getcwd()}")
    print("\n=== 2026年データ インポート ===")
    import_all()
    print("\n=== 完了 ===")
