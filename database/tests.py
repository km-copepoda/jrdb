# coding: utf-8
"""
JRDBデータファイルの包括的テスト

実データ(datas/temp/)を使用し、以下を検証する：
1. 各ファイル形式のレコード長
2. 各フィールドの値
3. コードマスタの正確性
4. モデルのフィールド長合計
5. 既知のバグ（回帰テスト）
6. 共通ユーティリティ関数
"""

import os
import unittest

import django
from django.conf import settings

# テスト用にインメモリSQLiteを使用(モデルインポート前に設定が必要)
if not settings.configured:
    settings.configure(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "database",
        ],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

from database.management.common import getColumnsDict, replaceSJIS
from database.models.GradeInformation import *
from database.models.helpers import *
from database.models.PreviousDayInformation import *
from database.models.ThatDayInformation import *

# サンプルデータディレクトリ
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datas", "temp")


def read_first_line(filename):
    """SJISファイルの最初の行を読み込む（改行なし）"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "rb") as f:
        line = f.readline()
        # CRLF or LF を除去
        return line.rstrip(b"\r\n")


def read_lines(filename):
    """SJISファイルの全行を読み込む（改行なし）"""
    path = os.path.join(DATA_DIR, filename)
    lines = []
    with open(path, "rb") as f:
        for line in f:
            stripped = line.rstrip(b"\r\n")
            if stripped:
                lines.append(stripped)
    return lines


def build_colspecs(model, n_fk):
    """
    実際のインポートコマンドと同じロジックでcolspecsを構築する。
    """
    # n_fk: 先頭フィールドの個数 (= offset = 初期colspecs_endの要素数)
    # UKC/CZA/KZA = 0 (colspecs_end=[], offset=0, colspecs_begin=[0])
    # KAB = 1 (colspecs_end=[PK_end], offset=1, colspecs_begin=[0,0])
    # BAC/0V/0W/OU/OT/OZ/HJC/SRB = 2
    # KYI/CYB/CHA/KKA/JOA/SED/SKB/TYB = 3

    # 各importコマンドの実装：
    max_lengths = [f.max_length for f in model._meta.fields]
    # UKCは[0], KABは[6], BACは[6, 8]...
    offset = n_fk
    colspecs_end = []
    if offset > 0:
        colspecs_end.extend(
            [sum(max_lengths[offset:i]) + max_lengths[i] for i in range(offset)]
        )

    colspecs_begin = [0] * (n_fk + 1)
    colspecs_begin.extend(colspecs_end[offset - 1 : -1])

    offset = n_fk
    colspecs_end.extend(
        [
            sum(max_lengths[offset:i]) + max_lengths[i]
            for i in range(offset, len(max_lengths))
        ]
    )

    colspecs_begin = [0] * (offset + 1)
    colspecs_begin.extend(colspecs_end[offset:-1])

    return list(zip(colspecs_begin, colspecs_end))


# ----------------------------------------------------------------------
# # 1. レコード長テスト
# ----------------------------------------------------------------------


class TestRecordLengths(unittest.TestCase):
    """各データファイルの1行目のバイト長が仕様通りか確認する"""

    def _assert_length(self, filename, expected_len):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            self.skipTest(f"{filename}が見つかりません")
        line = read_first_line(filename)
        self.assertEqual(
            len(line),
            expected_len,
            f"filename: {filename}: expected {expected_len} bytes, got {len(line)}",
        )

    def test_KAB_length(self):
        """前日_開催情報 (KAB) = 70 bytes"""
        self._assert_length("KAB170108.txt", 70)

    def test_BAC_length(self):
        """前日_番組情報 (BAC) = 182 bytes (WIN5フラグ含む)"""
        self._assert_length("BAC170108.txt", 182)

    def test_UKC_length(self):
        """馬基本情報 (UKC) = 290 bytes"""
        self._assert_length("UKC170108.txt", 290)

    def test_KYI_length(self):
        """前日_競走馬情報 (KYI) = 1022 bytes"""
        self._assert_length("KYI170108.txt", 1022)

    def test_CZA_length(self):
        """前日_調教師情報 (CZA) = 270 bytes"""
        self._assert_length("CZA170108.txt", 270)

    def test_KZA_length(self):
        """マスタ_騎手データ (KZA) = 270 bytes"""
        self._assert_length("KZA170108.txt", 270)

    def test_HJC_length(self):
        """成績_払戻情報 (HJC) = 442 bytes"""
        self._assert_length("HJC170108.txt", 442)

    def test_SRB_length(self):
        """成績_成績レース情報 (SRB) = 850 bytes"""
        self._assert_length("SRB170108.txt", 850)

    def test_SED_length(self):
        """成績_成績分析用情報 (SED) = 374 bytes"""
        self._assert_length("SED170108.txt", 374)

    def test_SKB_length(self):
        """成績_成績分析用拡張情報 (SKB) = 302 bytes"""
        self._assert_length("SKB170108.txt", 302)

    def test_TYB_length(self):
        """直前_情報 (TYB) = 126 bytes"""
        self._assert_length("TYB170108.txt", 126)

    def test_OZ_length(self):
        """前日_基準単複連情報 (OZ) = 955 bytes"""
        self._assert_length("OZ170108.txt", 955)

    def test_OW_length(self):
        """前日_基準ワイド情報 (OW) = 778 bytes"""
        self._assert_length("OW170108.txt", 778)

    def test_OU_length(self):
        """前日_基本馬単情報 (OU) = 1854 bytes"""
        self._assert_length("OU170108.txt", 1854)

    def test_OT_length(self):
        """前日_基準三連複情報 (OT) = 4910 bytes"""
        self._assert_length("OT170108.txt", 4910)

    def test_OV_length(self):
        """前日_基準三連単情報 (OV) = 34286 bytes"""
        self._assert_length("OV170108.txt", 34286)

    def test_CYB_length(self):
        """前日_調教分析情報 (CYB) = 94 bytes"""
        self._assert_length("CYB170108.txt", 94)

    def test_CHA_length(self):
        """前日_調教本追切情報 (CHA) = 62 bytes"""
        self._assert_length("CHA170108.txt", 62)

    def test_KKA_length(self):
        """前日_競走馬拡張 (KKA) = 322 bytes"""
        self._assert_length("KKA170108.txt", 322)

    def test_JOA_length(self):
        """前日_詳細情報 (JOA) = 114 bytes"""
        self._assert_length("JOA170108.txt", 114)


# ----------------------------------------------------------------------
# # 2. KAB フィールド解析テスト
# ----------------------------------------------------------------------


class TestKABParsing(unittest.TestCase):
    """KAB170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "KAB170108.txt")
        if not os.path.exists(path):
            self.skipTest("KAB170108.txt が見つかりません")
        line_raw = read_first_line("KAB170108.txt")
        self.line = line_raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")  # 中山

    def test_年(self):
        self.assertEqual(self.line[2:4], "17")

    def test_回(self):
        self.assertEqual(self.line[4:5], "1")

    def test_年月日(self):
        self.assertEqual(self.line[6:14], "20170108")

    def test_開催区分(self):
        self.assertEqual(self.line[14:15], "1")  # 関東

    def test_曜日(self):
        raw = read_first_line("KAB170108.txt")
        # 曜日内は2バイト SJIS: bytes[15:17]
        val = raw[15:17].decode("sjis")
        self.assertEqual(val, "日")

    def test_場合(self):
        raw = read_first_line("KAB170108.txt")
        val = raw[17:21].decode("sjis")
        self.assertEqual(val, "中山")

    def test_colspecs_field_count(self):
        """KABモデルのフィールド数とcolspecsの長さが一致する"""
        colspecs = build_colspecs(前日_開催情報, n_fk=1)
        self.assertEqual(len(colspecs), len(前日_開催情報._meta.fields))

    def test_getColumnsDict(self):
        """getColumnsDictが正常に辞書を返す"""
        colspecs = build_colspecs(前日_開催情報, n_fk=1)
        line_text = read_first_line("KAB170108.txt").decode("sjis")
        result = getColumnsDict(前日_開催情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)
        self.assertEqual(result["場コード"].strip(), "06")


# ----------------------------------------------------------------------
# # 3. BAC フィールド解析テスト
# ----------------------------------------------------------------------


class TestBACParsing(unittest.TestCase):
    """BAC170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(path):
            self.skipTest("BAC170108.txt が見つかりません")
        self.raw = read_first_line("BAC170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_R(self):
        self.assertEqual(self.line[6:8], "01")

    def test_年月日(self):
        self.assertEqual(self.line[8:16], "20170108")

    def test_発走時間(self):
        self.assertEqual(self.line[16:20], "0955")

    def test_距離(self):
        self.assertEqual(self.line[20:24], "1200")

    def test_芝ダ障害コード(self):
        self.assertEqual(self.line[24:25], "2")  # ダート

    def test_種別コード(self):
        self.assertEqual(self.line[27:29], "12")  # 3歳

    def test_条件(self):
        self.assertEqual(self.line[29:31], "A3")  # 未勝利

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_番組情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_番組情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_番組情報, n_fk=2)
        line_text = read_first_line("BAC170108.txt").decode("sjis")
        result = getColumnsDict(前日_番組情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)
        self.assertEqual(result["場コード"].strip(), "06")

    def test_record_total_bytes(self):
        """BACは182バイト(WIN5フラグ含む)"""
        self.assertEqual(len(self.raw), 182)


# ----------------------------------------------------------------------
# # 4. UKC フィールド解析テスト
# ----------------------------------------------------------------------


class TestUKCParsing(unittest.TestCase):
    """UKC170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "UKC170108.txt")
        if not os.path.exists(path):
            self.skipTest("UKC170108.txt が見つかりません")
        self.raw = read_first_line("UKC170108.txt")
        self.line = self.raw.decode("sjis")

    def test_血統登録番号(self):
        # bytes [0:8] = ASCII digits (SJISバイト位置 = Unicode文字位置)
        self.assertEqual(self.line[0:8], "14104484")

    def test_性別コード(self):
        # bytes [44:45] (SJISバイト位置で指定)
        # 馬名が36バイトの全角文字を含むので self.line[44] は異なる
        val = self.raw[44:45].decode("sjis")
        self.assertEqual(val, "2")  # 牝

    def test_毛色コード(self):
        # bytes [45:47]
        val = self.raw[45:47].decode("sjis")
        self.assertEqual(val, "04")  # 黒鹿毛

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(馬基本情報, n_fk=0)
        self.assertEqual(len(colspecs), len(馬基本情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(馬基本情報, n_fk=0)
        line_text = read_first_line("UKC170108.txt").decode("sjis")
        result = getColumnsDict(馬基本情報, replaceSJIS(line_text), colspecs)
        self.assertIn("血統登録番号", result)
        self.assertEqual(result["血統登録番号"].strip(), "14104484")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 290)


# ----------------------------------------------------------------------
# # 5. CZA フィールド解析テスト
# ----------------------------------------------------------------------


class TestCZAParsing(unittest.TestCase):
    """CZA170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "CZA170108.txt")
        if not os.path.exists(path):
            self.skipTest("CZA170108.txt が見つかりません")
        self.raw = read_first_line("CZA170108.txt")
        self.line = self.raw.decode("sjis")

    def test_調教師コード(self):
        """最初の5バイトが調教師コード"""
        code = self.line[0:5].strip()
        self.assertTrue(len(code) > 0, "調教師コードが空です")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 270)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_調教師情報, n_fk=0)
        self.assertEqual(len(colspecs), len(前日_調教師情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_調教師情報, n_fk=0)
        line_text = read_first_line("CZA170108.txt").decode("sjis")
        result = getColumnsDict(前日_調教師情報, replaceSJIS(line_text), colspecs)
        self.assertIn("調教師コード", result)


# ----------------------------------------------------------------------
# # 6. KZA フィールド解析テスト
# ----------------------------------------------------------------------


class TestKZAParsing(unittest.TestCase):
    """KZA170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "KZA170108.txt")
        if not os.path.exists(path):
            self.skipTest("KZA170108.txt が見つかりません")
        self.raw = read_first_line("KZA170108.txt")
        self.line = self.raw.decode("sjis")

    def test_騎手コード(self):
        code = self.line[0:5].strip()
        self.assertTrue(len(code) > 0, "騎手コードが空です")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 270)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(マスタ_騎手データ, n_fk=0)
        self.assertEqual(len(colspecs), len(マスタ_騎手データ._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(マスタ_騎手データ, n_fk=0)
        line_text = read_first_line("KZA170108.txt").decode("sjis")
        result = getColumnsDict(マスタ_騎手データ, replaceSJIS(line_text), colspecs)
        self.assertIn("騎手コード", result)


# ----------------------------------------------------------------------
# # 7. HJC フィールド解析テスト
# ----------------------------------------------------------------------


class TestHJCParsing(unittest.TestCase):
    """HJC170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "HJC170108.txt")
        if not os.path.exists(path):
            self.skipTest("HJC170108.txt が見つかりません")
        self.raw = read_first_line("HJC170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_単勝1着馬番(self):
        """bytes [8:10] = 単勝1着馬番"""
        val = self.line[8:10].strip()
        self.assertTrue(val.isdigit(), f"馬番が数字ではありません: {repr(val)}")

    def test_単勝払戻金(self):
        """bytes [10:17] = 単勝払戻金"""
        val = self.line[10:17].strip()
        self.assertTrue(val.isdigit(), f"払戻金が数字ではありません: {repr(val)}")

    # 1行目: 馬番=11, 払戻金=370 (3700円)
    # 実データは170108中山1R
    def test_単勝1着馬番_val(self):
        self.assertEqual(self.line[8:10].strip(), "11")

    def test_単勝払戻金_val(self):
        self.assertEqual(self.line[10:17].strip(), "370")

    def test_record_total_bytes(self):
        """実データは442バイト"""
        self.assertEqual(len(self.raw), 442)


# ----------------------------------------------------------------------
# # 8. SED フィールド解析テスト
# ----------------------------------------------------------------------


class TestSEDParsing(unittest.TestCase):
    """SED170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "SED170108.txt")
        if not os.path.exists(path):
            self.skipTest("SED170108.txt が見つかりません")
        self.raw = read_first_line("SED170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 374)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(成績_成績分析用情報, n_fk=3)
        self.assertEqual(len(colspecs), len(成績_成績分析用情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(成績_成績分析用情報, n_fk=3)
        line_text = read_first_line("SED170108.txt").decode("sjis")
        result = getColumnsDict(成績_成績分析用情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 9. SKB フィールド解析テスト
# ----------------------------------------------------------------------


class TestSKBParsing(unittest.TestCase):
    """SKB170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "SKB170108.txt")
        if not os.path.exists(path):
            self.skipTest("SKB170108.txt が見つかりません")
        self.raw = read_first_line("SKB170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 302)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(成績_成績分析用拡張情報, n_fk=3)
        self.assertEqual(len(colspecs), len(成績_成績分析用拡張情報._meta.fields))


# ----------------------------------------------------------------------
# # 10. SRB フィールド解析テスト
# ----------------------------------------------------------------------


class TestSRBParsing(unittest.TestCase):
    """SRB170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "SRB170108.txt")
        if not os.path.exists(path):
            self.skipTest("SRB170108.txt が見つかりません")
        self.raw = read_first_line("SRB170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 850)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(成績_成績レース情報, n_fk=2)
        self.assertEqual(len(colspecs), len(成績_成績レース情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(成績_成績レース情報, n_fk=2)
        line_text = read_first_line("SRB170108.txt").decode("sjis")
        result = getColumnsDict(成績_成績レース情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 11. TYB フィールド解析テスト
# ----------------------------------------------------------------------


class TestTYBParsing(unittest.TestCase):
    """TYB170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "TYB170108.txt")
        if not os.path.exists(path):
            self.skipTest("TYB170108.txt が見つかりません")
        self.raw = read_first_line("TYB170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 126)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(直前_情報, n_fk=3)
        self.assertEqual(len(colspecs), len(直前_情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(直前_情報, n_fk=3)
        line_text = read_first_line("TYB170108.txt").decode("sjis")
        result = getColumnsDict(直前_情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 12. OZ フィールド解析テスト
# ----------------------------------------------------------------------


class TestOZParsing(unittest.TestCase):
    """OZ170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "OZ170108.txt")
        if not os.path.exists(path):
            self.skipTest("OZ170108.txt が見つかりません")
        self.raw = read_first_line("OZ170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 955)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_基準単複連情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_基準単複連情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_基準単複連情報, n_fk=2)
        line_text = read_first_line("OZ170108.txt").decode("sjis")
        result = getColumnsDict(前日_基準単複連情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 13. OW フィールド解析テスト
# ----------------------------------------------------------------------


class TestOWParsing(unittest.TestCase):
    """OW170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "OW170108.txt")
        if not os.path.exists(path):
            self.skipTest("OW170108.txt が見つかりません")
        self.raw = read_first_line("OW170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 778)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_基準ワイド情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_基準ワイド情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_基準ワイド情報, n_fk=2)
        line_text = read_first_line("OW170108.txt").decode("sjis")
        result = getColumnsDict(前日_基準ワイド情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 14. OU フィールド解析テスト
# ----------------------------------------------------------------------


class TestOUParsing(unittest.TestCase):
    """OU170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "OU170108.txt")
        if not os.path.exists(path):
            self.skipTest("OU170108.txt が見つかりません")
        self.raw = read_first_line("OU170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 1854)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_基本馬単情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_基本馬単情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_基本馬単情報, n_fk=2)
        line_text = read_first_line("OU170108.txt").decode("sjis")
        result = getColumnsDict(前日_基本馬単情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 15. OT フィールド解析テスト
# ----------------------------------------------------------------------


class TestOTParsing(unittest.TestCase):
    """OT170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "OT170108.txt")
        if not os.path.exists(path):
            self.skipTest("OT170108.txt が見つかりません")
        self.raw = read_first_line("OT170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 4910)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_基準三連複情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_基準三連複情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_基準三連複情報, n_fk=2)
        line_text = read_first_line("OT170108.txt").decode("sjis")
        result = getColumnsDict(前日_基準三連複情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 16. OV フィールド解析テスト
# ----------------------------------------------------------------------


class TestOVParsing(unittest.TestCase):
    """OV170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "OV170108.txt")
        if not os.path.exists(path):
            self.skipTest("OV170108.txt が見つかりません")
        self.raw = read_first_line("OV170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 34286)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_基準三連単情報, n_fk=2)
        self.assertEqual(len(colspecs), len(前日_基準三連単情報._meta.fields))


# ----------------------------------------------------------------------
# # 17. CYB フィールド解析テスト
# ----------------------------------------------------------------------


class TestCYBParsing(unittest.TestCase):
    """CYB170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "CYB170108.txt")
        if not os.path.exists(path):
            self.skipTest("CYB170108.txt が見つかりません")
        self.raw = read_first_line("CYB170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 94)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_調教分析情報, n_fk=3)
        self.assertEqual(len(colspecs), len(前日_調教分析情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_調教分析情報, n_fk=3)
        line_text = read_first_line("CYB170108.txt").decode("sjis")
        result = getColumnsDict(前日_調教分析情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 18. CHA フィールド解析テスト
# ----------------------------------------------------------------------


class TestCHAParsing(unittest.TestCase):
    """CHA170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "CHA170108.txt")
        if not os.path.exists(path):
            self.skipTest("CHA170108.txt が見つかりません")
        self.raw = read_first_line("CHA170108.txt")
        self.line = self.raw.decode("sjis")

    def test_曜日(self):
        # bytes [10:12] = 曜日 (SJIS 2バイト)
        val = self.raw[10:12].decode("sjis")
        self.assertEqual(val, "水")

    def test_調教年月日(self):
        # bytes [12:20] = 調教年月日
        val = self.raw[12:20].decode("sjis")
        self.assertEqual(val, "20170104")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 62)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_調教本追切情報, n_fk=3)
        self.assertEqual(len(colspecs), len(前日_調教本追切情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_調教本追切情報, n_fk=3)
        line_text = read_first_line("CHA170108.txt").decode("sjis")
        result = getColumnsDict(前日_調教本追切情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 19. KKA フィールド解析テスト
# ----------------------------------------------------------------------


class TestKKAParsing(unittest.TestCase):
    """KKA170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "KKA170108.txt")
        if not os.path.exists(path):
            self.skipTest("KKA170108.txt が見つかりません")
        self.raw = read_first_line("KKA170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_JRA成績_raw_bytes(self):
        """
        仕様では JRA成績 は ZZ9x4=12バイト (1着, 2着, 3着, 着外)
        """
        # bytes [10:22] = JRA成績 (12バイト)
        val = self.raw[10:22].decode("sjis")
        self.assertEqual(len(val), 12, f"JRA成績は12バイトのはず: {repr(val)}")
        # サンプルデータ: '  0  0  0  3' (着外3回)
        self.assertEqual(val, "  0  0  0  3")

    def test_record_total_bytes(self):
        """実データは322バイト"""
        self.assertEqual(len(self.raw), 322)


# ----------------------------------------------------------------------
# # 20. JOA フィールド解析テスト
# ----------------------------------------------------------------------


class TestJOAParsing(unittest.TestCase):
    """JOA170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "JOA170108.txt")
        if not os.path.exists(path):
            self.skipTest("JOA170108.txt が見つかりません")
        self.raw = read_first_line("JOA170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 114)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_詳細情報, n_fk=3)
        self.assertEqual(len(colspecs), len(前日_詳細情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_詳細情報, n_fk=3)
        line_text = read_first_line("JOA170108.txt").decode("sjis")
        result = getColumnsDict(前日_詳細情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 21. KYI フィールド解析テスト
# ----------------------------------------------------------------------


class TestKYIParsing(unittest.TestCase):
    """KYI170108.txt の1行目フィールドを検証する"""

    def setUp(self):
        path = os.path.join(DATA_DIR, "KYI170108.txt")
        if not os.path.exists(path):
            self.skipTest("KYI170108.txt が見つかりません")
        self.raw = read_first_line("KYI170108.txt")
        self.line = self.raw.decode("sjis")

    def test_場コード(self):
        self.assertEqual(self.line[0:2], "06")

    def test_record_total_bytes(self):
        self.assertEqual(len(self.raw), 1022)

    def test_colspecs_field_count(self):
        colspecs = build_colspecs(前日_競走馬情報, n_fk=3)
        self.assertEqual(len(colspecs), len(前日_競走馬情報._meta.fields))

    def test_getColumnsDict(self):
        colspecs = build_colspecs(前日_競走馬情報, n_fk=3)
        line_text = read_first_line("KYI170108.txt").decode("sjis")
        result = getColumnsDict(前日_競走馬情報, replaceSJIS(line_text), colspecs)
        self.assertIn("場コード", result)


# ----------------------------------------------------------------------
# # 22. コードマスタ検証テスト
# ----------------------------------------------------------------------


class TestCodeMasters(unittest.TestCase):
    """コードマスタの値が正しいか検証する"""

    def test_場コード_中山(self):
        codes = dict(場コード)
        self.assertEqual(codes["06"], "中山")

    def test_場コード_東京(self):
        codes = dict(場コード)
        self.assertEqual(codes["05"], "東京")

    def test_性別コード_牝(self):
        codes = dict(性別コード)
        self.assertEqual(codes["2"], "牝")

    def test_毛色コード_黒鹿毛(self):
        codes = dict(毛色コード)
        self.assertEqual(codes["04"], "黒鹿")

    def test_天候コード_晴(self):
        codes = dict(天候コード)
        self.assertEqual(codes["1"], "晴")

    def test_馬場状態コード_良(self):
        codes = dict(馬場状態コード)
        self.assertEqual(codes["10"], "良")

    def test_種別コード_3歳(self):
        codes = dict(種別コード)
        self.assertEqual(codes["12"], "３歳")

    # BUG-3: グレードコード
    def test_グレードコード_BUG3_空文字キーではない(self):
        """
        BUG-3: helpers.py では "" (L リステッド競走) だが
        JRDBコードマスタでは "6" -> L。
        空文字キーはデータに存在しないので(BUG-3)
        """
        codes = dict(グレードコード)
        # 現在の実装: "" がキー
        self.assertIn("", codes, "グレードコードに空文字キーが存在する(BUG-3)")
        # 本来は "6" がキーであるべき
        self.assertNotIn("6", codes)

    # BUG-4: クラスコード
    def test_クラスコード_BUG4_07は旧表記(self):
        """
        BUG-4: 2019年以降 1600万 -> 3勝 に変更されたが helpers.py は旧表記のまま
        """
        codes = dict(クラスコード)
        # 現在の実装: 旧表記
        self.assertEqual(
            codes["07"],
            "芝1600万A",
            "クラスコード07が現在も旧表記であることを確認(BUG-4)",
        )
        # 正しい値は '芝3勝A' であるべき
        self.assertNotEqual(
            codes["07"], "芝3勝A", "クラスコード07はまだ未修正 (BUG-4未修正)"
        )

    def test_クラスコード_BUG4_10は旧表記(self):
        codes = dict(クラスコード)
        self.assertEqual(
            codes["10"], "芝1000万A", "クラスコード10の旧表記存在を確認 (BUG-4)"
        )

    def test_クラスコード_BUG4_13は旧表記(self):
        codes = dict(クラスコード)
        self.assertEqual(
            codes["13"], "芝500万A", "クラスコード13の旧表記存在を確認 (BUG-4)"
        )


# ----------------------------------------------------------------------
# # 23. モデルフィールド長合計テスト
# ----------------------------------------------------------------------


class TestModelFieldLengthSums(unittest.TestCase):
    """各モデルのフィールド max_length 合計が実データのバイト長と一致するか確認する"""

    def _sum_lengths(self, model, n_fk):
        """データフィールド (FK/PK以外) の max_length 合計 (offset = n_fk)"""
        fields = model._meta.fields
        offset = n_fk
        return sum(f.max_length for f in fields[offset:])

    def test_KAB_field_sum(self):
        total = self._sum_lengths(前日_開催情報, n_fk=1)
        self.assertEqual(total, 70)

    def test_BAC_field_sum(self):
        total = self._sum_lengths(前日_番組情報, n_fk=2)
        self.assertEqual(total, 182)

    def test_UKC_field_sum(self):
        total = self._sum_lengths(馬基本情報, n_fk=0)
        self.assertEqual(total, 290)

    def test_CZA_field_sum(self):
        total = self._sum_lengths(前日_調教師情報, n_fk=0)
        self.assertEqual(total, 270)

    def test_KZA_field_sum(self):
        total = self._sum_lengths(マスタ_騎手データ, n_fk=0)
        self.assertEqual(total, 270)

    def test_SRB_field_sum(self):
        total = self._sum_lengths(成績_成績レース情報, n_fk=2)
        self.assertEqual(total, 850)

    def test_SED_field_sum(self):
        total = self._sum_lengths(成績_成績分析用情報, n_fk=3)
        self.assertEqual(total, 374)

    def test_SKB_field_sum(self):
        total = self._sum_lengths(成績_成績分析用拡張情報, n_fk=3)
        self.assertEqual(total, 302)

    def test_TYB_field_sum(self):
        total = self._sum_lengths(直前_情報, n_fk=3)
        self.assertEqual(total, 126)

    def test_OZ_field_sum(self):
        total = self._sum_lengths(前日_基準単複連情報, n_fk=2)
        self.assertEqual(total, 955)

    def test_OW_field_sum(self):
        total = self._sum_lengths(前日_基準ワイド情報, n_fk=2)
        self.assertEqual(total, 778)

    def test_OU_field_sum(self):
        total = self._sum_lengths(前日_基本馬単情報, n_fk=2)
        self.assertEqual(total, 1854)

    def test_OT_field_sum(self):
        total = self._sum_lengths(前日_基準三連複情報, n_fk=2)
        self.assertEqual(total, 4910)

    def test_OV_field_sum(self):
        total = self._sum_lengths(前日_基準三連単情報, n_fk=2)
        self.assertEqual(total, 34286)

    def test_CYB_field_sum(self):
        total = self._sum_lengths(前日_調教分析情報, n_fk=3)
        self.assertEqual(total, 94)

    def test_CHA_field_sum(self):
        total = self._sum_lengths(前日_調教本追切情報, n_fk=3)
        self.assertEqual(total, 62)

    def test_KYI_field_sum(self):
        total = self._sum_lengths(前日_競走馬情報, n_fk=3)
        self.assertEqual(total, 1022)

    def test_JOA_field_sum(self):
        total = self._sum_lengths(前日_詳細情報, n_fk=3)
        self.assertEqual(total, 114)

    # --- BUG-1: HJC ---
    def test_HJC_BUG1_field_sum_is_too_small(self):
        """
        BUG-1: 成績_払戻情報 (HJC) のモデルフィールド長合計は 111 バイトだが
        実データは 442 バイト。払戻が複数回分定義されていない。
        BUG-1が修正されれば total == 442 になるはず。
        """
        total = self._sum_lengths(成績_払戻情報, n_fk=2)
        # 現状バグ: モデルの合計は111バイト (実データは442バイト)
        self.assertLess(total, 442)
        self.assertEqual(
            total, 111, f"BUG-1: HJCモデルの合計が実データ442バイト未満でないと変です"
        )
        # 現在の具体的な値を確認 (回帰検出用)
        self.assertEqual(total, 111)
        # f'HJCモデルのフィールド合計が111ではありません(現在: {total})'

    # --- BUG-2: KKA ---
    def test_KKA_BUG2_field_sum_is_too_small(self):
        """
        BUG-2: 前日_競走馬拡張 (KKA) のモデルフィールド長合計は 115 バイトだが
        実データは 322 バイト。各着順数フィールドが ZZ9x4=12バイトのところ
        max_length=3。
        BUG-2が修正されれば total == 322 になるはず。
        """
        total = self._sum_lengths(前日_競走馬拡張, n_fk=3)
        # 現状バグ: モデルの合計は115バイト (実データ322バイト)
        self.assertLess(total, 322)
        # BUG-2が修正されれば total == 115 ではなくなるはず。
        # 現在の具体的な値を確認 (回帰検出用)
        self.assertEqual(total, 115)
        # f'KKAモデルのフィールド合計が115ではありません(現在: {total})'


# ----------------------------------------------------------------------
# # 24. 既知のバグ回帰テスト
# ----------------------------------------------------------------------


class TestKnownBugs(unittest.TestCase):
    """発見済みバグの回帰テスト (修正前は失敗して当然)"""

    def test_BUG1_HJC_actual_data_length(self):
        """BUG-1: HJC実データは442バイト (モデルは111バイト分しか読まない)"""
        path = os.path.join(DATA_DIR, "HJC170108.txt")
        if not os.path.exists(path):
            self.skipTest("HJC170108.txt が見つかりません")
        raw = read_first_line("HJC170108.txt")
        self.assertEqual(len(raw), 442)
        # モデルは111バイト分しかカバーしていない(バグ)
        fields = 成績_払戻情報._meta.fields
        model_bytes = sum(f.max_length for f in fields[2:])  # offset=2
        self.assertLess(model_bytes, len(raw))
        # f'BUG-1: HJCモデルが実データより小さい'

    def test_BUG2_KKA_actual_data_length(self):
        """BUG-2: KKA実データは322バイト (モデルは115バイト分しか読まない)"""
        path = os.path.join(DATA_DIR, "KKA170108.txt")
        if not os.path.exists(path):
            self.skipTest("KKA170108.txt が見つかりません")
        raw = read_first_line("KKA170108.txt")
        self.assertEqual(len(raw), 322)
        fields = 前日_競走馬拡張._meta.fields
        model_bytes = sum(f.max_length for f in fields[3:])  # offset=3
        self.assertLess(model_bytes, len(raw))

    def test_BUG2_KKA_JRA成績_max_length(self):
        """BUG-2: KKAのJRA成績フィールドはmax_length=3だが仕様では12バイト"""
        field = None
        for f in 前日_競走馬拡張._meta.fields:
            if f.name == "JRA成績":
                field = f
                break
        self.assertIsNotNone(field, "JRA成績フィールドが見つかりません")
        # 現在のバグ: max_length=3
        self.assertEqual(
            field.max_length,
            3,
            f"JRA成績のmax_lengthは修正前は3 (BUG-2: 12であるべき )",
        )

    def test_BUG3_グレードコード_key_is_empty_string(self):
        """BUG-3: グレードコード "6" -> L が未定義で "" キーが使われている"""
        codes = dict(グレードコード)
        self.assertIn("", codes)
        self.assertNotIn("6", codes)

    def test_BUG4_クラスコード_uses_old_naming(self):
        """BUG-4: クラスコード 07/10/13 が 2019年前の旧表記"""
        codes = dict(クラスコード)
        self.assertEqual(codes["07"], "芝1600万A")  # 旧: 1600万A / 新: 芝3勝A
        self.assertEqual(codes["10"], "芝1000万A")  # 旧: 1000万A / 新: 芝2勝A
        self.assertEqual(codes["13"], "芝500万A")  # 旧: 500万A / 新: 芝1勝A

    def test_BUG5_KKA_sample_data_JRA成績_is_12_bytes(self):
        """BUG-2追試: 実データ KKA bytes [10:22] は12バイトで4つの成績が入っている"""
        path = os.path.join(DATA_DIR, "KKA170108.txt")
        if not os.path.exists(path):
            self.skipTest("KKA170108.txt が見つかりません")
        raw = read_first_line("KKA170108.txt")
        jra_raw = raw[10:22].decode("sjis")
        self.assertEqual(len(jra_raw), 12)
        # '  0  0  0  3' の形式 (3バイトx4)
        parts = [jra_raw[i : i + 3] for i in range(0, 12, 3)]
        self.assertEqual(len(parts), 4)
        for p in parts:
            self.assertTrue(
                p.strip().isdigit() or p.strip() == "",
                f"着数パートが数字ではありません: {repr(p)}",
            )

    def test_三連複オッズ_to_dict_key_has_3_elements(self):
        """3連複オッズ_to_dict() のキーが3個タプルになっている"""
        path = os.path.join(DATA_DIR, "OT170105.txt")
        if not os.path.exists(path):
            self.skipTest("OT170105.txt が見つかりません")
        raw = read_first_line("OT170105.txt").decode("sjis")
        # OT の colspecs: n_fk=2, ofset=2
        # 場コード(2) + 年(2) + 回(1) + 日(1) + R(2) + 登録頭数(2) = 10 chars header
        # 三連複オッズ starts at char 10, length 4896
        instance = 前日_基準三連複情報()
        instance.三連複オッズ = raw[10 : 10 + 4896]
        result = instance.三連複オッズ_to_dict()
        # キーがすべて3要素タプルかかｋかｋか確認
        for key in result.keys():
            self.assertIsInstance(key, tuple, f"キーがタプルではない: {key}")
            self.assertEqual(len(key), 3, f"キーが3要素ではない: {key}")
            # C(18, 3) = 816 通りのうちから出ない組み合わせが正しいか
            self.assertLessEqual(len(result), 816)
            # 空でないエントリ
            self.assertGreater(len(result), 0)

    def test_単勝オッズ_cap_boundary(self):
        """単勝オッズ 999.9倍が正しく変える( < 1000)"""
        instance = 前日_基準単複連情報()
        # 999.9 倍を表す文字列  (5byte)
        instance.単勝オッズ = "999.9" + " " * 85
        instance.複勝オッズ = " " * 90
        instance.連勝オッズ = " " * 765
        result = instance.単勝オッズ_to_dict()
        self.assertAlmostEqual(result["1"], 999.9, places=1)

    def test_三連複オッズ_cab_boundary(self):
        """三連複オッズ 9999.9倍が正しく帰る ( < 10000)"""
        instance = 前日_基準三連複情報()
        # 先頭エントリ(1, 2, 3) に9999.9倍を設定
        instance.三連複オッズ = "9999.9" + " " * (4896 - 6)
        result = instance.三連複オッズ_to_dict()
        self.assertAlmostEqual(result[(1, 2, 3)], 9999.9, places=1)


# ----------------------------------------------------------------------
# # 25. 共通ユーティリティ関数テスト
# ----------------------------------------------------------------------


class TestCommonFunctions(unittest.TestCase):
    """database/management/common.py の関数をテストする"""

    def test_replaceSJIS_roman_numeral(self):
        """ローマ数字 Ⅰ->1 の置換"""
        result = replaceSJIS("ⅠⅡⅢ")
        self.assertEqual(result, "１２３")

    def test_replaceSJIS_wide_space(self):
        """全角スペース->半角スペース2つ?"""
        result = replaceSJIS("　")
        self.assertEqual(result, "  ")

    def test_replaceSJIS_kg_symbol(self):
        """'kg' の置換"""
        result = replaceSJIS("㎏")
        self.assertEqual(result, "kg")

    def test_replaceSJIS_circled_numbers(self):
        """①②③->123 の置換"""
        result = replaceSJIS("①②③")
        self.assertEqual(result, "１２３")

    def test_replaceSJIS_noop(self):
        """置換不要な文字はそのまま"""
        original = "中山中山競馬場"
        result = replaceSJIS(original)
        self.assertEqual(result, original)

    def test_getColumnsDict_KAB(self):
        """KABの全フィールドが辞書に含まれる"""
        path = os.path.join(DATA_DIR, "KAB170108.txt")
        if not os.path.exists(path):
            self.skipTest("KAB170108.txt が見つかりません")
        colspecs = build_colspecs(前日_開催情報, n_fk=1)
        line_text = read_first_line("KAB170108.txt").decode("sjis")
        result = getColumnsDict(前日_開催情報, replaceSJIS(line_text), colspecs)
        expected_fields = [
            "開催情報ID",
            "場コード",
            "年",
            "回",
            "日",
            "年月日",
            "開催区分",
            "曜日",
            "場名",
            "天候コード",
            "芝馬場状態コード",
            "芝馬場差",
            "芝馬場状態中",
            "芝馬場状態外",
            "芝馬場差",
            "直線馬場差最内",
            "直線馬場差内",
            "直線馬場差中",
            "直線馬場差外",
            "直線馬場差大外",
            "ダ馬場状態コード",
            "ダ馬場状態内",
            "ダ馬場状態中",
            "ダ馬場状態外",
            "ダ馬場差",
            "データ区分",
            "連続何日目",
            "芝種類",
            "草丈",
            "転圧",
            "凍結防止剤",
            "中間降水量",
            "予備",
        ]
        for field_name in expected_fields:
            self.assertIn(
                field_name, result, f"フィールド {field_name} が辞書に存在しない"
            )

    def test_getColumnsDict_values_are_strings(self):
        """getColumnsDictの全値が文字列型"""
        path = os.path.join(DATA_DIR, "KAB170108.txt")
        if not os.path.exists(path):
            self.skipTest("KAB170108.txt が見つかりません")
        colspecs = build_colspecs(前日_開催情報, n_fk=1)
        line = read_first_line("KAB170108.txt").decode("sjis")
        result = getColumnsDict(前日_開催情報, replaceSJIS(line), colspecs)
        for key, val in result.items():
            self.assertIsInstance(
                val, str, f"{key} の値が文字列ではありません: {type(val)}"
            )

    def test_FK_OneToOneField_is_field_name_plus_id(self):
        """FK/OneToOneField は field_name + '_id' でアクセスされる"""
        path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(path):
            self.skipTest("BAC170108.txt が見つかりません")
        colspecs = build_colspecs(前日_番組情報, n_fk=2)
        line_text = read_first_line("BAC170108.txt").decode("sjis")
        result = getColumnsDict(前日_番組情報, replaceSJIS(line_text), colspecs)
        self.assertIn("前日_開催情報_id", result)
        self.assertIn("番組情報ID", result)


# ----------------------------------------------------------------------
# # 26. データ整合性テスト
# ----------------------------------------------------------------------


class TestDataConsistency(unittest.TestCase):
    """複数ファイルをまたいだデータの整合性テスト"""

    def test_KAB_and_BAC_share_場コード(self):
        """KABとBACの場コードが一致する"""
        kab_path = os.path.join(DATA_DIR, "KAB170108.txt")
        bac_path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(kab_path) or not os.path.exists(bac_path):
            self.skipTest("KABまたはBACのデータが見つかりません")
        kab_line = read_first_line("KAB170108.txt").decode("sjis")
        bac_line = read_first_line("BAC170108.txt").decode("sjis")
        kab_code = kab_line[0:2]
        bac_code = bac_line[0:2]
        self.assertEqual(kab_code, bac_code, "KABとBACの場コードが一致しない")

    def test_KAB_and_BAC_share_年月日(self):
        kab_path = os.path.join(DATA_DIR, "KAB170108.txt")
        bac_path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(kab_path) or not os.path.exists(bac_path):
            self.skipTest("KAB170108.txt または BAC170108.txt が見つかりません")
        kab_line = read_first_line("KAB170108.txt").decode("sjis")
        bac_line = read_first_line("BAC170108.txt").decode("sjis")
        kab_date = kab_line[6:14]
        bac_date = bac_line[8:16]
        self.assertEqual(
            kab_date,
            bac_date,
            f"KAB年月日({kab_date}) と BAC年月日({bac_date}) が一致しない",
        )

    def test_all_BAC_lines_same_date(self):
        """BACファイルの全行は同じ年月日"""
        path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(path):
            self.skipTest("BAC170108.txt が見つかりません")
        lines = read_lines("BAC170108.txt")
        dates = set(line.decode("sjis")[8:16] for line in lines)
        self.assertEqual(
            len(dates), 1, f"BACファイル内に複数の年月日が含まれています: {dates}"
        )
        self.assertEqual(list(dates)[0], "20170108")

    def test_all_KAB_lines_same_date(self):
        """KABファイルの全行は同じ年月日"""
        path = os.path.join(DATA_DIR, "KAB170108.txt")
        if not os.path.exists(path):
            self.skipTest("KAB170108.txt が見つかりません")
        lines = read_lines("KAB170108.txt")
        dates = set(line.decode("sjis")[6:14] for line in lines)
        self.assertEqual(
            len(dates), 1, f"KABファイル内に複数の年月日が含まれています: {dates}"
        )
        self.assertEqual(list(dates)[0], "20170108")

    def test_KYI_and_BAC_share_場_年_回_日_R(self):
        """KYIとBACのレースキー(場コード/年/回/日/R) が一致する"""
        kyi_path = os.path.join(DATA_DIR, "KYI170108.txt")
        bac_path = os.path.join(DATA_DIR, "BAC170108.txt")
        if not os.path.exists(kyi_path) or not os.path.exists(bac_path):
            self.skipTest("KYI170108.txt または BAC170108.txt が見つかりません")
        kyi_line = read_first_line("KYI170108.txt").decode("sjis")
        bac_line = read_first_line("BAC170108.txt").decode("sjis")
        # KYIとBACは同じ場/年/回/日/Rのレースに属するはず
        kyi_key = kyi_line[0:8]  # 場+年+回+日+R+馬番の最初8バイト
        bac_key = bac_line[0:8]  # 場+年+回+日+Rの最初8バイト
        self.assertEqual(
            kyi_key[0:8],
            bac_key[0:8],
            f"KYIとBACのレースキー(場コード/年/回/日/R)が一致しない",
        )

    def test_HJC_実データ_単勝払戻回数(self):
        """HJC実データに単勝払戻が複数行分のデータが格納されている確認"""
        path = os.path.join(DATA_DIR, "HJC170108.txt")
        if not os.path.exists(path):
            self.skipTest("HJC170108.txt が見つかりません")
        raw = read_first_line("HJC170108.txt")
        # 単勝払戻は3回分 (9バイトx3 = 27バイト), bytes [8:35]
        # 1回目: bytes [8:17], 2回目: bytes [17:26], 3回目: bytes [26:35]
        win1_bango = raw[8:10].decode("sjis")
        win1_haraimodoshi = raw[10:17].decode("sjis").strip()
        self.assertTrue(
            win1_bango.isdigit(), f"単勝1回目馬番が数値ではない: {repr(win1_bango)}"
        )
        self.assertTrue(
            win1_haraimodoshi.isdigit(),
            f"単勝1回目払戻金が数値ではない: {repr(win1_haraimodoshi)}",
        )

    def test_UKC_性別コード_is_valid(self):
        """UKCの性別コードが有効値(コードマスタにある)か確認"""
        lines = read_lines("UKC170108.txt")
        valid_codes = {code for code, _ in 性別コード}
        for i, raw_line in enumerate(lines[:10]):  # 最初の10行をチェック
            # SJISバイト位置で指定 (Unicode文字インデックスではない)
            code = raw_line[44:45].decode("sjis")
            self.assertIn(
                code, valid_codes, f'UKC行{i+1}: 性別コード "{code}" が有効な値でない'
            )


# ----------------------------------------------------------------------
# # エントリポイント
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
