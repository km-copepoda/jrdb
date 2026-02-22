"""DB接続設定と定数定義"""
import os

# PostgreSQL 接続設定（環境変数 or デフォルト値）
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ.get("DB_NAME", "jrdb"),
    "user": os.environ.get("DB_USER", "admin"),
    "password": os.environ.get("DB_PASS", "admin"),
}

# 荒れたレース判定: 確定単勝人気順位・着順の上位3位とみなす値
# JRDBデータは固定幅テキストなので、空白パディングのバリエーションがある
TOP3_VALUES = ("1", "2", "3", "01", "02", "03", " 1", " 2", " 3")

# テーブル名（Django の app_label + model名）
T_BAC = "database_前日_番組情報"
T_KAB = "database_前日_開催情報"
T_KYI = "database_前日_競走馬情報"
T_SED = "database_成績_成績分析用情報"
T_UKC = "database_馬基本情報"
T_KZA = "database_マスタ_騎手データ"
T_CZA = "database_前日_調教師情報"
T_KKA = "database_前日_競走馬拡張"
T_CYB = "database_前日_調教分析情報"
T_TYB = "database_直前_情報"
T_JOA = "database_前日_詳細情報"
T_HJC = "database_成績_払戻情報"
T_CHK = "database_前日_調教本追切情報"
T_SRJ = "database_成績_成績レース情報"

# KYI の数値指数フィールド（レース単位で集約する対象）
KYI_NUMERIC_FIELDS = [
    "IDM", "騎手指数", "情報指数", "総合指数", "人気指数",
    "調教指数", "厩舎指数", "基準オッズ", "基準複勝オッズ",
    "テン指数", "ペース指数", "上がり指数", "位置指数",
    "激走指数", "万券指数",
]

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
