"""
DB設定とテーブル定義
"""

DB_CONFIG = {
    "dbname": "jrdb",
    "user": "admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432",
}

# テーブル名 (Django 自動生成 → db_table で指定した名前)
TABLES = {
    "BAC": "database_race_info",
    "KYI": "database_pre_race_info",
    "CYB": "database_workout_info",
    "CHA": "database_change_info",
    "KKA": "database_thisday_info",
    "JOA": "database_prediction_info",
    "SED": "database_result_sed",
    "HUC": "database_result_huc",
    "SRB": "database_result_srb",
    "SKB": "database_result_skb",
    "TYB": "database_tyb_info",
    "KAB": "database_kai_info",
    "UKC": "database_horse",
    "CZA": "database_trainer",
    "KZA": "database_jockey",
    "OZ":  "database_win_place_odds",
    "OT":  "database_trio_odds",
    "OU":  "database_quinella_odds",
    "OV":  "database_trifecta_odds",
    "OW":  "database_wide_odds",
}

# LightGBM ハイパーパラメータ (Optuna チューニング済み)
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
BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_jobs": -1,
    "verbosity": -1,
}

# シード一覧 (7-seed ensemble)
SEEDS = [42, 49, 56, 63, 70, 77, 84]
