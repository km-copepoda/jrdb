"""
特徴量パイプライン (137特徴量)
build_horse_features() が 1行=1出走馬 の DataFrame を構築する。
"""

import pandas as pd
import numpy as np
import psycopg2
from config import DB_CONFIG, TABLES


def _get_connection():
    return psycopg2.connect(**DB_CONFIG)


def _safe_float(series):
    """文字列カラムを float に変換。変換不能は NaN。"""
    return pd.to_numeric(series.str.strip(), errors="coerce")


def build_horse_features():
    """
    基礎テーブル結合 (SQL) で特徴量 DataFrame を構築する。
    1行 = 1出走馬。

    結合:
        KYI (前走情報) をベースに
        BAC (レース情報), KAB (開催情報), SRB (成績レース情報),
        SED (成績分析用), CYB (調教), JOA (本紙予想), TYB (直前情報)
    を結合。
    """
    conn = _get_connection()

    sql = f"""
    SELECT
        -- 基本ID
        kyi."競走馬情報ID"                   AS horse_race_id,
        kyi."前日_番組情報_id"               AS race_id,
        bac."年月日"                         AS date,
        kyi."馬基本情報_id"                  AS horse_id,
        kyi."馬番"                           AS umaban,

        -- 目的変数 (top3_finish: 3着以内=1)
        sed."着順"                           AS finish_pos,
        sed."確定単勝人気順位"               AS final_pop,
        sed."確定単勝オッズ"                 AS final_odds,

        -- KYI (22項目): 各指数
        kyi."IDM"                            AS kyi_idm,
        kyi."騎手指数"                       AS kyi_jockey_idx,
        kyi."情報指数"                       AS kyi_info_idx,
        kyi."総合指数"                       AS kyi_total_idx,
        kyi."人気指数"                       AS kyi_pop_idx,
        kyi."調教指数"                       AS kyi_train_idx,
        kyi."厩舎指数"                       AS kyi_stable_idx,
        kyi."基準オッズ"                     AS kyi_base_odds,
        kyi."基準人気順位"                   AS kyi_base_pop,
        kyi."基準複勝オッズ"                 AS kyi_base_place_odds,
        kyi."基準複勝人気順位"               AS kyi_base_place_pop,
        kyi."テン指数"                       AS kyi_ten_idx,
        kyi."ペース指数"                     AS kyi_pace_idx,
        kyi."上がり指数"                     AS kyi_agari_idx,
        kyi."位置指数"                       AS kyi_position_idx,
        kyi."激走指数"                       AS kyi_gekisou_idx,
        kyi."脚質コード"                     AS kyi_running_style,
        kyi."距離適性コード"                 AS kyi_distance_apt,
        kyi."上昇度"                         AS kyi_joushou,
        kyi."騎手期待連対率"                 AS kyi_jockey_rentai,
        kyi."騎手期待単勝率"                 AS kyi_jockey_win_rate,
        kyi."騎手期待3着内率"                AS kyi_jockey_top3_rate,

        -- BAC (レース条件 9項目)
        bac."距離"                           AS bac_distance,
        bac."芝ダ障害コード"                 AS bac_track_type,
        bac."グレードコード"                 AS bac_grade,
        bac."条件"                           AS bac_condition,
        bac."種別コード"                     AS bac_class,
        bac."重量"                           AS bac_weight_rule,
        bac."頭数"                           AS bac_horse_count,
        bac."発走時間"                       AS bac_start_time,
        bac."場コード"                       AS bac_venue,

        -- KAB (開催情報 5項目)
        kab."天候コード"                     AS kab_weather,
        kab."芝馬場状態コード"               AS kab_turf_condition,
        kab."ダ馬場状態コード"               AS kab_dirt_condition,
        kab."芝馬場差"                       AS kab_turf_diff,
        kab."ダ馬場差"                       AS kab_dirt_diff,

        -- CYB (調教 8項目)
        cyb."追切指数"                       AS cyb_oikiri_idx,
        cyb."仕上指数"                       AS cyb_shiage_idx,
        cyb."調教量評価"                     AS cyb_volume_eval,
        cyb."仕上指数変化"                   AS cyb_shiage_change,
        cyb."調教評価"                       AS cyb_train_eval,
        cyb."調教タイプ"                     AS cyb_train_type,
        cyb."調教コース種別"                 AS cyb_course_type,
        cyb."調教距離"                       AS cyb_train_distance,

        -- JOA (本紙予想 10項目)
        joa."基準オッズ"                     AS joa_base_odds,
        joa."基準複勝オッズ"                 AS joa_base_place_odds,
        joa."CID調教素点"                    AS joa_cid_train,
        joa."CID厩舎素点"                    AS joa_cid_stable,
        joa."CID素点"                        AS joa_cid_raw,
        joa."CID"                            AS joa_cid,
        joa."LS指数"                         AS joa_ls_idx,
        joa."LS評価"                         AS joa_ls_eval,
        joa."EM"                             AS joa_em,
        joa."厩舎BB印"                       AS joa_stable_bb,

        -- TYB (直前情報 10項目)
        tyb."IDM"                            AS tyb_idm,
        tyb."騎手指数"                       AS tyb_jockey_idx,
        tyb."情報指数"                       AS tyb_info_idx,
        tyb."オッズ指数"                     AS tyb_odds_idx,
        tyb."パドック指数"                   AS tyb_paddock_idx,
        tyb."総合指数"                       AS tyb_total_idx,
        tyb."単勝オッズ"                     AS tyb_win_odds,
        tyb."複勝オッズ"                     AS tyb_place_odds,
        tyb."馬体重"                         AS tyb_weight,
        tyb."馬体重増減"                     AS tyb_weight_diff,

        -- SED (成績 17項目)
        sed."IDM"                            AS sed_idm,
        sed."素点"                           AS sed_raw_score,
        sed."テン指数"                       AS sed_ten_idx,
        sed."上がり指数"                     AS sed_agari_idx,
        sed."ペース指数"                     AS sed_pace_idx,
        sed."馬体重"                         AS sed_weight,
        sed."馬体重増減"                     AS sed_weight_diff,
        sed."コーナー順位1"                  AS sed_corner1,
        sed."コーナー順位2"                  AS sed_corner2,
        sed."コーナー順位3"                  AS sed_corner3,
        sed."コーナー順位4"                  AS sed_corner4,
        sed."脚質コード"                     AS sed_running_style,
        sed."馬ペース"                       AS sed_horse_pace,
        sed."レースペース"                   AS sed_race_pace,
        sed."前3Fタイム"                     AS sed_first_3f,
        sed."後3Fタイム"                     AS sed_last_3f,
        sed."タイム"                         AS sed_time

    FROM {TABLES['KYI']} AS kyi
    JOIN {TABLES['BAC']} AS bac
        ON kyi."前日_番組情報_id" = bac."番組情報ID"
    JOIN {TABLES['KAB']} AS kab
        ON bac."前日_開催情報_id" = kab."開催情報ID"
    LEFT JOIN {TABLES['SED']} AS sed
        ON kyi."競走馬情報ID" = sed."前日_競走馬情報_id"
    LEFT JOIN {TABLES['CYB']} AS cyb
        ON kyi."競走馬情報ID" = cyb."前日_競走馬情報_id"
    LEFT JOIN {TABLES['JOA']} AS joa
        ON kyi."競走馬情報ID" = joa."前日_競走馬情報_id"
    LEFT JOIN {TABLES['TYB']} AS tyb
        ON kyi."競走馬情報ID" = tyb."前日_競走馬情報_id"
    WHERE sed."着順" IS NOT NULL
    ORDER BY bac."年月日", bac."場コード", bac."R", kyi."馬番"
    """

    df = pd.read_sql(sql, conn)
    conn.close()

    # --- 数値変換 ---
    numeric_cols = [
        "kyi_idm", "kyi_jockey_idx", "kyi_info_idx", "kyi_total_idx",
        "kyi_pop_idx", "kyi_train_idx", "kyi_stable_idx",
        "kyi_base_odds", "kyi_base_pop", "kyi_base_place_odds", "kyi_base_place_pop",
        "kyi_ten_idx", "kyi_pace_idx", "kyi_agari_idx", "kyi_position_idx",
        "kyi_gekisou_idx", "kyi_jockey_rentai",
        "kyi_jockey_win_rate", "kyi_jockey_top3_rate",
        "bac_distance", "bac_horse_count",
        "kab_turf_diff", "kab_dirt_diff",
        "cyb_oikiri_idx", "cyb_shiage_idx",
        "joa_base_odds", "joa_base_place_odds",
        "joa_cid_train", "joa_cid_stable", "joa_cid_raw", "joa_cid",
        "joa_ls_idx",
        "tyb_idm", "tyb_jockey_idx", "tyb_info_idx",
        "tyb_odds_idx", "tyb_paddock_idx", "tyb_total_idx",
        "tyb_win_odds", "tyb_place_odds",
        "tyb_weight", "tyb_weight_diff",
        "sed_idm", "sed_raw_score", "sed_ten_idx", "sed_agari_idx", "sed_pace_idx",
        "sed_weight", "sed_weight_diff",
        "sed_corner1", "sed_corner2", "sed_corner3", "sed_corner4",
        "sed_first_3f", "sed_last_3f", "sed_time",
        "final_pop", "final_odds", "finish_pos",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    # --- 目的変数 ---
    df["top3_finish"] = (df["finish_pos"] <= 3).astype(int)
    df["win"] = (df["finish_pos"] == 1).astype(int)

    # --- 年カラム ---
    df["year"] = df["date"].astype(str).str[:4].astype(int)

    # --- 派生特徴量 ---
    df = _add_derived_features(df)

    print(f"build_horse_features: {len(df)} rows, {len(df.columns)} columns")
    return df


def _add_derived_features(df):
    """派生特徴量を追加"""

    # race_idm_rank: レース内のIDM順位
    df["race_idm_rank"] = df.groupby("race_id")["kyi_idm"].rank(
        ascending=False, method="min"
    )

    # idm_diff: レース平均IDMとの差
    race_mean_idm = df.groupby("race_id")["kyi_idm"].transform("mean")
    df["idm_diff"] = df["kyi_idm"] - race_mean_idm

    # odds_change: 基準人気 vs 確定人気 (人気変動)
    df["odds_change"] = df["kyi_base_pop"] - df["final_pop"]

    # prev1_finish: 前走着順 (同一馬の直前レースの着順)
    df = df.sort_values(["horse_id", "date"])
    df["prev1_finish"] = df.groupby("horse_id")["finish_pos"].shift(1)
    df["prev1_top3"] = (df["prev1_finish"] <= 3).astype(float)
    df["prev1_top3"] = df["prev1_top3"].where(df["prev1_finish"].notna(), np.nan)

    # avg_finish_5: 直近5走の平均着順
    df["avg_finish_5"] = (
        df.groupby("horse_id")["finish_pos"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # mean_prev1_finish: レース内の前走平均着順
    df["mean_prev1_finish"] = df.groupby("race_id")["prev1_finish"].transform("mean")

    # レース内の当日オッズ順位
    df["tyb_odds_rank"] = df.groupby("race_id")["tyb_win_odds"].rank(
        ascending=True, method="min"
    )

    # レース内の総合指数順位
    df["total_idx_rank"] = df.groupby("race_id")["kyi_total_idx"].rank(
        ascending=False, method="min"
    )

    # date順に戻す
    df = df.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)

    return df


# 特徴量カラムのリスト (モデル学習に使うカラム)
FEATURE_COLS = [
    # KYI (22)
    "kyi_idm", "kyi_jockey_idx", "kyi_info_idx", "kyi_total_idx",
    "kyi_pop_idx", "kyi_train_idx", "kyi_stable_idx",
    "kyi_base_odds", "kyi_base_pop", "kyi_base_place_odds", "kyi_base_place_pop",
    "kyi_ten_idx", "kyi_pace_idx", "kyi_agari_idx", "kyi_position_idx",
    "kyi_gekisou_idx", "kyi_jockey_rentai",
    "kyi_jockey_win_rate", "kyi_jockey_top3_rate",
    "kyi_running_style", "kyi_distance_apt", "kyi_joushou",
    # BAC+KAB (13)
    "bac_distance", "bac_track_type", "bac_grade", "bac_condition",
    "bac_class", "bac_weight_rule", "bac_horse_count", "bac_venue",
    "kab_weather", "kab_turf_condition", "kab_dirt_condition",
    "kab_turf_diff", "kab_dirt_diff",
    # CYB (8)
    "cyb_oikiri_idx", "cyb_shiage_idx", "cyb_volume_eval",
    "cyb_shiage_change", "cyb_train_eval", "cyb_train_type",
    "cyb_course_type", "cyb_train_distance",
    # JOA (10)
    "joa_base_odds", "joa_base_place_odds",
    "joa_cid_train", "joa_cid_stable", "joa_cid_raw", "joa_cid",
    "joa_ls_idx", "joa_ls_eval", "joa_em", "joa_stable_bb",
    # TYB (10)
    "tyb_idm", "tyb_jockey_idx", "tyb_info_idx",
    "tyb_odds_idx", "tyb_paddock_idx", "tyb_total_idx",
    "tyb_win_odds", "tyb_place_odds", "tyb_weight", "tyb_weight_diff",
    # SED (17)
    "sed_idm", "sed_raw_score", "sed_ten_idx", "sed_agari_idx", "sed_pace_idx",
    "sed_weight", "sed_weight_diff",
    "sed_corner1", "sed_corner2", "sed_corner3", "sed_corner4",
    "sed_running_style", "sed_horse_pace", "sed_race_pace",
    "sed_first_3f", "sed_last_3f", "sed_time",
    # 派生特徴量 (8)
    "race_idm_rank", "idm_diff", "odds_change",
    "prev1_finish", "prev1_top3", "avg_finish_5",
    "mean_prev1_finish", "tyb_odds_rank",
]

# カテゴリカルカラム (label encoding 対象)
CATEGORICAL_COLS = [
    "bac_track_type", "bac_grade", "bac_condition", "bac_class",
    "bac_weight_rule", "bac_venue",
    "kab_weather", "kab_turf_condition", "kab_dirt_condition",
    "kyi_running_style", "kyi_distance_apt", "kyi_joushou",
    "cyb_volume_eval", "cyb_shiage_change", "cyb_train_eval",
    "cyb_train_type", "cyb_course_type", "cyb_train_distance",
    "joa_ls_eval", "joa_em", "joa_stable_bb",
    "sed_running_style", "sed_horse_pace", "sed_race_pace",
]


def prepare(df):
    """ラベルエンコーディング等の前処理"""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    # カテゴリカル変数のラベルエンコーディング
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            mask = df[col].notna()
            df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
            df[col] = pd.to_numeric(df[col], errors="coerce")
            label_encoders[col] = le

    # 使用する特徴量カラムが存在するか確認
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} features missing: {missing[:5]}...")

    print(f"prepare: {len(available)} features available")
    return df, available, label_encoders
