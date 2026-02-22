import os
import io
import zipfile
import requests
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from database.management.common import valid_year_month_day, date_range, getModel, getColumnsDict
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from django.db.models.fields.related import ManyToOneRel
from django.db.utils import OperationalError
import time

# JRDB ダウンロード設定
JRDB_BASE_URL = "https://jrdb.com/member/datazip"

# JRDB のデータコード → ディレクトリ名 / zip内ファイルコード のマッピング
# key: import コマンドが期待するファイル接頭辞 (例: "CZA")
# value: (JRDB ディレクトリ名, JRDB zip/ファイル接頭辞)
JRDB_CODE_MAP = {
    "BAC": ("Bac", "BAC"),
    "CHA": ("Cha", "CHA"),
    "CYB": ("Cyb", "CYB"),
    "CZA": ("Cs",  "CSA"),   # JRDB は CSA、ローカルは CZA
    "HJC": ("Hjc", "HJC"),
    "JOA": ("Jo",  "JOA"),
    "KAB": ("Kab", "KAB"),
    "KKA": ("Kka", "KKA"),
    "KYI": ("Kyi", "KYI"),
    "KZA": ("Ks",  "KSA"),   # JRDB は KSA、ローカルは KZA
    "OT":  ("Ot",  "OT"),
    "OU":  ("Ou",  "OU"),
    "OV":  ("Ov",  "OV"),
    "OW":  ("Ow",  "OW"),
    "OZ":  ("Oz",  "OZ"),
    "SED": ("Sed", "SED"),
    "SKB": ("Skb", "SKB"),
    "TYB": ("Tyb", "TYB"),
    "UKC": ("Ukc", "UKC"),
}


def _get_jrdb_auth():
    """環境変数から JRDB 認証情報を取得"""
    user = os.environ.get("JRDB_USER", "")
    password = os.environ.get("JRDB_PASS", "")
    if user and password:
        return (user, password)
    return None


def _download_from_jrdb(local_code, date_str, dest_path):
    """
    JRDB から zip ファイルをダウンロードして txt を展開する。

    Args:
        local_code: ローカルファイルのコード (例: "CZA", "SED")
        date_str: 日付文字列 YYMMDD (例: "170108")
        dest_path: 保存先パス (例: "./database/temp/CZA170108.txt")
    Returns:
        True if successful, False otherwise
    """
    auth = _get_jrdb_auth()
    if not auth:
        print("Warning: JRDB_USER / JRDB_PASS が未設定のためダウンロードをスキップ")
        return False

    if local_code not in JRDB_CODE_MAP:
        print(f"Warning: {local_code} の JRDB マッピングが未定義")
        return False

    jrdb_dir, jrdb_code = JRDB_CODE_MAP[local_code]
    zip_name = f"{jrdb_code}{date_str}.zip"

    # JRDB の URL: datazip/{Dir}/{20XX}/{CODE}{YYMMDD}.zip
    year_4digit = "20" + date_str[:2]
    url = f"{JRDB_BASE_URL}/{jrdb_dir}/{year_4digit}/{zip_name}"

    try:
        print(f"Downloading {url} ...")
        response = requests.get(url, auth=auth, timeout=60)
        response.raise_for_status()

        # zip を展開
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # zip 内のファイル名を探す (1ファイルのはず)
            names = zf.namelist()
            txt_name = None
            for name in names:
                if name.lower().endswith(".txt"):
                    txt_name = name
                    break
            if not txt_name:
                print(f"Warning: {zip_name} 内に txt ファイルが見つかりません: {names}")
                return False

            # 展開して保存 (JRDB のファイル名がローカルと異なる場合はリネーム)
            data = zf.read(txt_name)
            with open(dest_path, "wb") as f:
                f.write(data)
            print(f"Saved to {dest_path}")
            return True

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # 開催日でない日はファイルが存在しない（正常）
            pass
        else:
            print(f"Warning: Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"Warning: Failed to download {url}: {e}")
        return False


class Command(BaseCommand):
    help = "This command can not use!"

    # サブクラスで設定する属性:
    #   file_format: ローカルファイルパスのフォーマット (例: './database/temp/SED{date}.txt')
    #   jrdb_code:   JRDB データコード (例: 'SED')。JRDB_CODE_MAP のキーに対応。
    #                未設定の場合は file_format から自動推定する。

    def add_arguments(self, parser):
        parser.add_argument(
            "-begin",
            "--find_begin_date",
            type=valid_year_month_day,
            help="YYMMDD or YYYYMMDD",
        )
        parser.add_argument(
            "-end",
            "--find_end_date",
            type=valid_year_month_day,
            help="YYMMDD or YYYYMMDD",
        )
        parser.add_argument(
            "--force-fetch",
            action="store_true",
            help="Force download even if local file exists",
        )
        parser.add_argument(
            "--no-download",
            action="store_true",
            help="Skip downloading, only import local files",
        )

    def _get_jrdb_code(self):
        """JRDB データコードを取得（jrdb_code 属性 or file_format から推定）"""
        if hasattr(self, "jrdb_code"):
            return self.jrdb_code
        # file_format から推定: './database/temp/SED{date}.txt' → 'SED'
        basename = os.path.basename(self.file_format)
        code = basename.split("{")[0]
        return code

    def handle(self, *args, **options):
        force_fetch = options.get("force_fetch", False)
        no_download = options.get("no_download", False) or os.environ.get("SKIP_DOWNLOAD", "") == "1"
        begin_date = (
            options["find_begin_date"]
            if options["find_begin_date"]
            else dt.now().date()
        )
        end_date = (
            options["find_end_date"] if options["find_end_date"] else dt.now().date()
        )

        start_time = time.time()
        os.makedirs("./database/temp", exist_ok=True)

        jrdb_code = self._get_jrdb_code()

        # ファイルをダウンロード（必要な場合）
        for file_date in date_range(begin_date, end_date):
            date_str = file_date.strftime("%y%m%d")
            filename = self.file_format.format(date=date_str)
            file_exists = os.path.isfile(filename)

            # ローカルにない または 強制ダウンロード指定の場合、ダウンロードを試みる
            if not no_download and (not file_exists or force_fetch):
                _download_from_jrdb(jrdb_code, date_str, filename)

        # ローカルファイルから読み込み
        for file_date in date_range(begin_date, end_date):
            filename = self.file_format.format(date=file_date.strftime("%y%m%d"))
            if not os.path.isfile(filename):
                continue
            read_and_write(filename, self.__model, self.__colspecs)

        end_time = time.time() - start_time
        print(end_time)



def read_and_write(filename, my_model, colspecs):
    print(f"import file is START -> {filename}")
    create_models, update_models = read_file(filename, my_model, colspecs)
    write_db(create_models, update_models, my_model)
    print(f"import file is END -> {filename}")


def read_file(filename, my_model, colspecs):
    with open(filename, "r", encoding="cp932") as f:
        models = list(map(lambda line: getModel(my_model, line, colspecs), f))
    create_models = []
    update_models = []
    for m in models:
        if my_model.objects.filter(pk=m.pk).exists():
            update_models.append(m)
        else:
            create_models.append(m)
    return create_models, update_models


def write_db(create_models, update_models, my_model):
    try:
        with transaction.atomic():
            my_model.objects.bulk_create(create_models)
            for m in update_models:
                m.save()
    except Exception as e:
        print(f"DB write error (skipped): {e}")
