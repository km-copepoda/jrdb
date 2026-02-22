"""
特徴量パイプライン (137特徴量)
build_horse_features() が 1行=1出走馬 の DataFrame を構築する。
"""

import pandas as pd
import psycopg2
from config import DB_CONFIG, TABLES


def _get_connection():
    return psycopg2.connect(**DB_CONFIG)


def build_horse_features():
    """
    基礎テーブル結合 (SQL) で特徴量 DataFrame を構築する。

    SELECT
        kyi.血統登録番号 AS horse_race_id,
        kyi.開催ID AS race_id,
        bac.年月日 AS date,
        -- 目的変数
        CASE WHEN srb.確定着順 IN ('1','2','3') THEN 1 ELSE 0 END AS top3_finish,
        -- KYI (22項目): IDM, 騎手指数, 情報指数, ...
        -- BAC+KAB (13項目): 距離, 芝ダ, 天候, 競馬場, ...
        -- CYB (8項目): 調教F, 調教ランク, 調教評価, ...
        -- JOA/TYB (10項目): オッズ, パドック点, 馬体, 気配, ...
        -- SED (ハロン, タイム差, 通過順, 上り3F, ...)
    FROM {kyi} AS kyi
    JOIN {bac} AS bac ON kyi.開催ID = bac.開催ID
    JOIN {srb} AS srb ON kyi.開催ID = srb.開催ID AND kyi.血統登録番号 = srb.血統登録番号
    LEFT JOIN {sed} AS sed ON ...

    派生特徴量:
        - prev1_finish, prev1_top3_finish
        - avg_finish_5, mean_prev1_finish
        - race_idm_rank: レース内のIDM順位
        - idm_diff: レース平均IDMとの差
        - odds_change: 前日人気 vs 当日人気
    """
    # TODO: SQL クエリを実装し DataFrame を返す
    raise NotImplementedError("build_horse_features() は未実装です")


def prepare(df):
    """ラベルエンコーディング等の前処理"""
    # TODO: カテゴリカル変数のエンコーディング
    raise NotImplementedError("prepare() は未実装です")
