JRDB競馬予測システム - 再構築プロンプト
このドキュメントは、本システムを別マシンでゼロから再構築するための完全な仕様書である。
外部データ（JRDBダウンロードファイル、DB中身）は含まない。コードとアーキテクチャのみ。

1. 全体アーキテクチャ
jrdb/
├── docker-compose.yml     # DB + Backend + Frontend の統合起動
├── .env                   # JRDB認証情報
├── .gitignore
├── backend/               # Django REST API + JRDBデータインポート
│   ├── Dockerfile
│   ├── entrypoint.sh      # DB待機 -> migrate -> import -> runserver
│   ├── requirements.txt
│   ├── manage.py
│   ├── jrdb/              # Django project設定
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── database/          # JRDBデータモデル + インポートコマンド
│   │   ├── models.py
│   │   └── management/
│   │       └── commands/
│   │           ├── _PreviousDayInformation.py # 前日系テーブル (KYI, BAC, KAB等)
│   │           ├── _ThisDayInformation.py     # 直前情報 (TYB)
│   │           ├── _GradeInformation.py       # 成績系テーブル (SED, HUC, SRB等)
│   │           └── _import_BaseCommand.py     # JRDB zip DL + 固定幅パース + DB書込
│   ├── common_py          # CP932変換、colspan計算
│   ├── ml_training/       # # import_BAC, import_SED... 各テーブル
│   │   ├── temp/          # ダウンロードした txt ファイル (gitignore対象)
│   │   ├── horse_model/   # 競馬予測MLエンジン
│   │   └── commands/      # ...
│   └── horse/             # 競馬REST API
│       ├── serializers.py # filters.py
│       ├── views.py       # REST API (レース一覧, CSV出力)
│       └── urls.py
└── frontend/              # React + Vite SPA
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── api.js         # /api/ml/〜 へのfetch
        └── App.jsx        # ML本体 (ローカル実行)
2. インフラ構成 (Docker Compose)
docker-compose.yml
YAML
services:
  db:
    image: postgres:15
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
      POSTGRES_DB: jrdb

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      DB_HOST: db
      DB_NAME: jrdb
      DB_USER: admin
      DB_PASS: admin
      ALLOWED_HOSTS: "*" "localhost 127.0.0.1 backend"
      IMPORT_START: "20180101"
      IMPORT_END: "20241231"
      JRDB_USER: "JRDBユーザー名" # ローカルファイルがある場合
      JRDB_PASS: "JRDBパスワード"
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      API_URL: "http://backend:8000"
    depends_on:
      - backend

volumes:
  db_data:
Backend Dockerfile
Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
Backend requirements.txt
django
pandas
psycopg2-binary
djangorestframework
django-filter
requests
beautifulsoup4
lxml
scikit-learn
lightgbm
optuna
matplotlib
seaborn
Frontend Dockerfile
Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
entrypoint.sh
Bash
#!/bin/bash
# 1. PostgreSQL 接続を待機 (psycopg2でリトライ)
# 2. python manage.py migrate
# 3. JRDBデータインポートコマンド実行 (翌月〜末日で日付範囲指定) :
#   - python manage.py import_PreviousDayInformation --UKC, CZA, KZA
#   - python manage.py import_PreviousDayInformation --KAB, BAC, KYI, CYB, CHA, KKA, JOA, OZ, OT, OU, OV, OW
#   - python manage.py import_GradeInformation --HUC, SRB, SKB, SED
#   - python manage.py import_ThisDayInformation --TYB
# 4. python manage.py runserver 0.0.0.0:8000
3. データベーススキーマ (Django Models)
各ファイルコードがDjangoモデルに1:1対応。

主要テーブル一覧
ファイル | Djangoモデル | テーブル名 | 説明 | PK

UKC | 馬基本 | database_horse | 馬基本情報 (血統登録番号, 馬名, 父母) | 血統登録番号(ID)

CZA | 調教師 | database_trainer | 調教師マスター (成績, リーディング) | 調教師コード(ID)

KZA | 騎手 | database_jockey | 騎手マスター (成績, リーディング) | 騎手コード(ID)

KAB | 開催 | database_kai_info | 開催情報 (レース開催日, 距離, 芝ダ, グレード, 競馬場) | 開催ID (年月日+競馬場コード)

BAC | レース | database_race_info | レース基本 (レース名, 頭数, 賞金, 前走脚質傾向) | 開催ID+レース番号

KYI | 前走 | database_pre_race_info | 前走・前々走・近走詳細 (前走着順, 斤量, 通過順, 上り3F, 位置取り) | 血統登録番号+開催ID+レース番号

CYB | 調教 | database_workout_info | 調教内容 (ウッド, 坂路, 併せ, 強さ, 時計, 調教評価) | 前記

CHA | 変更 | database_change_info | 変更情報 (テン乗り, 斤量変更, 初ブリンカー, 去勢) | 前記

KKA | 直前 | database_thisday_info | 当日馬体重, 馬場状態, 本紙予想, 血統適性 | 前記

JOA | 本紙 | database_prediction_info | 本紙オッズ・パドック情報 (印, 調教評価, 返し馬評価) | 前記

TYB | 直前情報 | database_tyb_info | 当日オッズ, パドック点数, 馬体変化 | 前記

OZ | 単複 | database_win_place_odds | 単複オッズ (時系列) | 前記

HUC | 成績 | database_result_huc | レース結果詳細 (上位入着, 返金, 荒れ判定指標) | 開催ID+レース番号

SRB | 成績馬 | database_result_srb | 各馬成績詳細 (1着〜最下位, 上り3Fタイム, 位置取り, 通過順, 差) | 血統登録番号+開催ID+レース番号

SKB | 成績分析 | database_result_skb | 競馬・騎手・血統別成績、馬場状態別分析結果 | パドック評価、コメント含む

SED | 成績拡張 | database_result_sed | JRDB独自成績 (ペース, 各ハロンタイム, 3F前後半, 各馬上がり3F, 通過順詳細, コメント) | 血統登録番号+開催ID+レース番号

OT | 三連複 | database_trio_odds | 三連複オッズ | 開催ID+レース番号

OU | 馬連 | database_quinella_odds | 馬連オッズ | 開催ID+レース番号

OV | ワイド | database_wide_odds | ワイドオッズ | 開催ID+レース番号

OW | 三連単 | database_trifecta_odds | 三連単オッズ | 開催ID+レース番号

データインポートの仕組み
Import_BaseCommand.py が共通基盤:

JRDB_CODE_MAP: ローカルコード <-> JRDBディレクトリ/ZIPのコーディング

例: "CZA" -> ("CZA", "CZA"), "KZA" -> ("KZA", "KZA")

ダウンロード: https://jrdb.com/member/data/Pkydata/index.html (要認証)

Basic認証 (JRDB_USER/JRDB_PASS) で対応

パース: common_py の getColumnsDict() で仕様読み込み。

各行を固定長テキストファイル (CP932) としてデコードしてDjangoモデルに変換

_import_XXX.py で個別変換ロジックを実装

bulk_create で高速保存 (1,000行/10秒程度)

Django設定 (jrdb/settings.py)
Engine: postgresql_psycopg2

DB接続: DB_HOST, DB_NAME, DB_USER, DB_PASS

REST Framework: PageNumberPagination, PAGE_SIZE=10, DjangoFilterBackend

INSTALLED_APPS: database, horse, ml, rest_framework, django_filters

URL構成
/api/horses -> HorseViewSet (GET list/detail, 馬名フィルタあり)

/api/races -> RaceViewSet (レース一覧, 荒れ判定付き)

/api/fields -> FieldListView (ML用フィールド一覧)

/api/csv/download -> CsvDownloadView (馬単位のCSV出力)

/api/ml/predict -> PredictView (予測実行)

/ -> React SPA (index.html)

4. MLパイプライン (★システムの核心)
4.1 概要
目的: 競馬レースの「荒れ」を予測し、回収率が高い馬を抽出する

予測モデル: fav1, fav2, fav3 (当日オッズ上位3頭) が全員3着以内に入らないレース (発生率 〜33%)

手法: LightGBM 7-seed ensemble, target = P(win) = 内その馬が1着になる確率

ローリング評価: 2018-N-1年で学習 -> N+1年を予測 (2024, 2025, 2026)

4.2 環境セットアップ
Bash
cd ml_training
pip install -r requirements.txt
# (pandas, numpy, scipy, scikit-learn, lightgbm, psycopg2-binary, matplotlib, optuna)
4.3 config.py -- DB設定とテーブル定義
Python
DB_CONFIG = {
    "dbname": "jrdb", "user": "admin", "password": "admin",
    "host": "localhost", "port": "5432"
}
# テーブル名 (Django自動生成)
TABLES = {
    "BAC": "database_race_info", "KYI": "database_pre_race_info",
    "CYB": "database_workout_info", "SED": "database_result_sed",
    "HUC": "database_result_huc", "TYB": "database_tyb_info"
}
4.4 horse_model.py -- 特徴量パイプライン (137特徴量)
build_horse_features() が1行=1出走馬のDataFrameを構築。

基礎テーブル結合 (SQL)
SQL
SELECT
    kyi.血統登録番号 AS horse_race_id,
    kyi.開催ID AS race_id,
    bac.年月日 AS date,
    -- 目的変数
    CASE WHEN SRB.確定着順 IN ('1','2','3') THEN 1 ELSE 0 END AS top3_finish,
    -- 独自分析
    sed.確定単勝人気 AS final_pop,
    kyi.IDM AS idm, kyi.騎手指数, kyi.情報指数, ... (計22項目)
    -- BAC+KAB(レース条件) (13項目)
    -- CYB(調教) (8項目)
    -- JOA(本紙予想) (10項目)
FROM database_pre_race_info AS kyi
JOIN database_race_info AS bac ON kyi.開催ID = bac.開催ID
JOIN database_result_srb AS srb ON kyi.開催ID = srb.開催ID AND kyi.血統登録番号 = srb.血統登録番号
LEFT JOIN database_result_sed ON ...
特徴量統合 (137項目)
BAC/KAB: 距離, 芝ダ, 天候, 競馬場, 等

KYI: IDM, 騎手, 情報, 展開, 激走, 予想, 各指数

CYB: 調教F, 調教ランク, 調教評価

JOA/TYB: オッズ, パドック点, 馬体, 気配

SED: 各ハロン, タイム差, 通過順, 上り3F

派生特徴量
prev1_finish - prev1_top3_finish

avg_finish_5, mean_prev1_finish

race_idm_rank: レース内のIDM順位

idm_diff: レース平均IDMとの差

odds_change: 前日人気 vs 当日人気

4.5 LightGBM ハイパーパラメータ
Python
TUNED_PARAMS = {
    "learning_rate": 0.078,
    "num_leaves": 92,
    "max_depth": 11,
    "min_child_samples": 34,
    "feature_fraction": 0.54,
    "bagging_fraction": 0.61,
    "lambda_l1": 1e-07,
    "lambda_l2": 5.09e-06,
}
# 共通パラメータ
"objective": "binary",
"metric": "auc",
"scale_pos_weight": neg/pos, # クラス不均衡対策
"n_jobs": -1,
"verbosity": -1,
4.6 upset_ef.py -- モデル評価パイプライン
処理フロー
build_horse_features() -> prepare() (ラベルエンコーディング)

train(): LightGBM 7-seed ensemble (seed: 42, 49, 56, ... 84)

build_race_features(): レース単位で集計

当日オッズ上位3頭を fav1, fav2, fav3 として抽出

p_threat: 4番人気以下の馬の勝率合計 sum(P(win))

fav_win_prob: fav1, 2, 3 が3着ともに入らない確率

upset_score: 実績 vs 予測の乖離 (荒れ度合い)

compute_scores():

naive: P_product

odds_naive: 1 / odds

blend50: 0.5 * naive + 0.5 * odds_naive

rest_threat: p_threat / sum(P_win)

4.7 calc_e_tickets.py -- 券種別ROIグリッドサーチ
対象券種 (12種)
馬連 (全通り) | Cin1_2j*100 ! なし !

馬連 (ex2) | 1, 2番人気除外

馬連 (ex2_1av) | 1, 2番人気が1頭だけ入る

単勝 (全通り) | P1j*100 ! なし !

三連複 (全通り) | Pin3-3j*300 ! なし !

三連複 (ex1) | 1番人気除外

三連複 (ex2) | 1, 2番人気除外

三連複 (ex3) | 1, 2, 3番人気除外

ベストパフォーマンス (2024-2026ローリング実績)
馬連 ex2 | rest_threat top3% | r_nh16 h14 f4.0: 167.4% (+75.2万円)

特徴: 高ROI, 安定型 (購入額小)

三連複 ex2 | 利益最大化 | rest_threat top3%: 161.7% (+56.1万円)

特徴: 利益総額最大, 資金力が必要

ex2+fav1_2av1 がベストバランス

7. データインポート手順
初回セットアップ
Bash
# 1. Docker起動
docker compose up -d
# DB初期化を待つ (30秒)

# 2. Backend起動 (migrate + import)
docker compose up backend
# JRDB_DOWNLOAD=1 ならJRDBから自動ダウンロード
# またはローカルに txt ファイルを置いて SKIP_DOWNLOAD=1
追加データインポート (例: 2026年)
import_2026.py を実行

JRDBから全テーブルの zip をダウンロード -> txt 展開

Django management command で DB インポート

8. 実行手順
Bash
# 特徴量生成・学習
cd ml_training
python train.py

# 全券種の利益を計算
python calc_e_tickets.py

# モデル評価パイプライン (グリッドサーチ)
python upset_ef.py
9. コード改善メモ
既知のバグ
PreviousDayInformation.py L341: float(value) -> float(value) の typo (修正済)。

注意点
JRDBのデータはCP932固定長。common_py の replaceSUSU() で機種依存文字を変換

同一馬の転厩、馬主変更に注意 (血統登録番号が主キー)

レースIDは KAB 依存 (場コード, 年, 月, 回, 日, レース番号)

BEGIN/END を変更して実行

import_2026.py を実行

train.py

calc_e_tickets.py

upset_ef.py