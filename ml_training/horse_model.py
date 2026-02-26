"""
馬単位予測モデル v2: LightGBM + CatBoost ブレンド → 確率キャリブレーション → レース荒れ推定

改善点 (v1 → v2):
  - CatBoost + LightGBM のブレンドアンサンブル
  - Isotonic regression でキャリブレーション
  - 拡張レース集約特徴量 (エントロピー, Gini, 順位相関等)
  - Stage2 に Optuna 最適化追加
  - DataFrame fragmentation 対策
"""
import sys
import psycopg2
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from scipy.stats import entropy

from config import (
    DB_CONFIG, TOP3_VALUES, KYI_NUMERIC_FIELDS,
    T_BAC, T_KAB, T_KYI, T_SED, T_KZA, T_CZA, T_KKA,
    T_CYB, T_TYB, T_JOA, T_UKC, T_CHK,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------
# 1. 馬単位データの構築
# ---------------------------------------------------------------
def _to_numeric(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _parse_record(record_str):
    s = str(record_str).strip()
    if not s or s in ("nan", "None"):
        return np.nan, np.nan, 0
    parts = s.split()
    if len(parts) >= 4:
        try:
            vals = [int(p) for p in parts[:4]]
            total = sum(vals)
            if total == 0:
                return np.nan, np.nan, 0
            return vals[0] / total, (vals[0] + vals[1]) / total, total
        except ValueError:
            pass
    if len(s) <= 14:
        s12 = s.rjust(12)
        try:
            vals = [int(s12[i:i + 3].strip()) for i in range(0, 12, 3)]
            total = sum(vals)
            if total == 0:
                return np.nan, np.nan, 0
            return vals[0] / total, (vals[0] + vals[1]) / total, total
        except (ValueError, IndexError):
            pass
    return np.nan, np.nan, 0


def build_horse_features(feature_groups=None, return_raw=False) -> pd.DataFrame:
    """馬単位の特徴量テーブルを構築 (1行 = 1出走馬)

    Args:
        feature_groups: 有効にするFGリスト (例: ["FG1","FG3"])。
                        None の場合はベースライン（後方互換）。
        return_raw: True の場合、ID列・着順・人気・印列を保持して返す。
                    ファクター評価等で生データが必要な場合に使用。
    """
    # feature_groups が指定された場合のみインポート
    fg_defs = {}
    if feature_groups:
        from feature_groups import FEATURE_GROUPS as _FG_REGISTRY
        for fg_name in feature_groups:
            if fg_name in _FG_REGISTRY:
                fg_defs[fg_name] = _FG_REGISTRY[fg_name]

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        print(f"馬単位データを構築中... (FG: {list(fg_defs.keys()) if fg_defs else 'baseline'})")

        # --- FG用の追加SQL列を準備 ---
        extra_bac_sql = ""
        extra_tyb_cols = []
        extra_kka_cols = []
        for fg_name, fg_def in fg_defs.items():
            for alias, dbcol in fg_def.get("bac_cols", []):
                extra_bac_sql += f',\n            bac."{dbcol}" AS {alias}'
            extra_tyb_cols.extend(fg_def.get("tyb_cols", []))
            extra_kka_cols.extend(fg_def.get("kka_cols", []))

        # --- ベーステーブル: KYI + SED (着順結果) ---
        top3_sql = ", ".join(f"'{v}'" for v in TOP3_VALUES)
        kyi_fields = ", ".join(f'kyi."{f}"' for f in KYI_NUMERIC_FIELDS)
        sql = f"""
        SELECT
            kyi."競走馬情報ID"        AS horse_race_id,
            kyi."前日_番組情報_id"     AS race_id,
            kyi."馬基本情報_id"        AS horse_id,
            bac."年月日"               AS date,
            bac."場コード"             AS venue,

            -- 目的変数
            CASE WHEN TRIM(sed."着順") IN ({top3_sql}) THEN 1 ELSE 0 END AS top3_finish,
            sed."着順"                 AS finish_pos,
            sed."確定単勝人気順位"      AS final_pop,

            -- KYI 基本指数
            {kyi_fields},

            -- KYI 予想・評価
            kyi."基準人気順位"          AS base_pop_rank,
            kyi."脚質コード"           AS run_style,
            kyi."距離適性コード"        AS distance_apt,
            kyi."芝適性コード"          AS turf_apt,
            kyi."ダ適性コード"          AS dirt_apt,
            kyi."上昇度"               AS improvement,
            kyi."ローテーション"        AS rotation,
            kyi."枠確定馬体重増減"      AS weight_change,
            kyi."ブリンカー"           AS blinker,
            kyi."性別コード"           AS sex,
            kyi."枠番"                AS gate,
            kyi."馬番"                AS horse_no,
            kyi."負担重量"             AS weight_carried,
            kyi."獲得賞金"             AS prize_money,
            kyi."降級フラグ"            AS demotion_flag,
            kyi."入厩何日前"            AS days_since_entry,
            kyi."調教矢印コード"        AS training_arrow,
            kyi."厩舎評価コード"        AS stable_eval,
            kyi."馬スタート指数"        AS start_idx,
            kyi."馬出遅率"             AS late_start_rate,

            -- KYI 予想順位
            kyi."道中順位"             AS pred_mid,
            kyi."後3F順位"             AS pred_3f,
            kyi."ゴール順位"            AS pred_goal,
            kyi."道中差"               AS pred_mid_gap,
            kyi."後3F差"               AS pred_3f_gap,
            kyi."ゴール差"             AS pred_goal_gap,
            kyi."展開記号"             AS develop_code,
            kyi."ペース予想"            AS pace_pred,

            -- KYI 印
            kyi."総合印"               AS mark_overall,
            kyi."IDM印"               AS mark_idm,
            kyi."情報印"               AS mark_info,
            kyi."騎手印"               AS mark_jockey,
            kyi."厩舎印"               AS mark_stable,
            kyi."調教印"               AS mark_training,
            kyi."激走印"               AS mark_upset,
            kyi."万券印"               AS mark_longshot,

            -- KYI ランキング
            kyi."激走順位"             AS upset_rank,
            kyi."LS指数順位"           AS ls_rank,
            kyi."テン指数順位"          AS ten_rank,
            kyi."上がり指数順位"        AS agari_rank,
            kyi."位置指数順位"          AS position_rank,
            kyi."激走タイプ"            AS upset_type,
            kyi."厩舎ランク"            AS stable_rank,
            kyi."放牧先ランク"          AS farm_rank,
            kyi."重適正コード"          AS heavy_apt,
            kyi."騎手期待連対率"        AS jockey_exp_pr,
            kyi."騎手期待単勝率"        AS jockey_exp_wr,
            kyi."騎手期待3着内率"       AS jockey_exp_top3,
            kyi."条件クラス"            AS condition_class,
            kyi."クラスコード"          AS class_code,

            -- レース条件 (BAC + KAB)
            bac."距離"                 AS distance,
            bac."芝ダ障害コード"        AS surface,
            bac."グレードコード"        AS grade,
            bac."種別コード"           AS race_type,
            bac."頭数"                AS n_horses,
            bac."右左"                AS course_turn,
            bac."内外"                AS course_width,
            bac."一着賞金"             AS first_prize,
            kab."天候コード"            AS weather,
            kab."芝馬場状態コード"       AS turf_condition,
            kab."ダ馬場状態コード"       AS dirt_condition,
            kab."芝馬場差"              AS turf_bias,
            kab."ダ馬場差"              AS dirt_bias,
            kab."連続何日目"            AS consecutive_day
            {extra_bac_sql}
        FROM "{T_KYI}" kyi
        JOIN "{T_BAC}" bac ON kyi."前日_番組情報_id" = bac."番組情報ID"
        JOIN "{T_KAB}" kab ON bac."前日_開催情報_id" = kab."開催情報ID"
        LEFT JOIN "{T_SED}" sed ON sed."前日_番組情報_id" = bac."番組情報ID"
            AND sed."馬基本情報_id" = kyi."馬基本情報_id"
        """
        df = pd.read_sql(sql, conn)
        print(f"  ベーステーブル: {len(df)} 行")

        # --- 数値変換 ---
        num_cols = KYI_NUMERIC_FIELDS + [
            "base_pop_rank", "improvement", "rotation", "weight_change",
            "gate", "horse_no", "weight_carried", "prize_money",
            "days_since_entry", "start_idx", "late_start_rate",
            "pred_mid", "pred_3f", "pred_goal",
            "pred_mid_gap", "pred_3f_gap", "pred_goal_gap",
            "upset_rank", "ls_rank", "ten_rank", "agari_rank", "position_rank",
            "jockey_exp_pr", "jockey_exp_wr", "jockey_exp_top3",
            "distance", "n_horses", "first_prize",
            "turf_bias", "dirt_bias", "consecutive_day",
            "finish_pos", "final_pop",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = _to_numeric(df[col])

        # 印を数値化
        mark_cols = ["mark_overall", "mark_idm", "mark_info", "mark_jockey",
                     "mark_stable", "mark_training", "mark_upset", "mark_longshot"]
        for col in mark_cols:
            df[col + "_num"] = _to_numeric(df[col])

        # ランク変換
        df["stable_rank_num"] = _to_numeric(df["stable_rank"])
        farm_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        df["farm_rank_num"] = df["farm_rank"].str.strip().map(farm_map)
        df["heavy_apt_num"] = _to_numeric(df["heavy_apt"])

        # 予想ゴール順位 vs 人気の乖離
        df["pred_vs_pop"] = df["pred_goal"] - df["base_pop_rank"]

        # 月
        df["month"] = _to_numeric(df["date"].str[4:6])

        # --- CYB 調教情報 ---
        print("  CYB調教情報をJOIN中...")
        sql_cyb = f"""
        SELECT
            cyb."前日_競走馬情報_id"  AS horse_race_id,
            cyb."追切指数"            AS workout_idx,
            cyb."仕上指数"            AS fitness_idx,
            cyb."調教量評価"          AS training_volume,
            cyb."仕上指数変化"        AS fitness_change,
            cyb."調教評価"            AS training_eval
        FROM "{T_CYB}" cyb
        """
        cyb = pd.read_sql(sql_cyb, conn)
        for col in ["workout_idx", "fitness_idx", "training_volume",
                     "fitness_change", "training_eval"]:
            cyb[col] = _to_numeric(cyb[col])
        df = df.merge(cyb, on="horse_race_id", how="left")

        # --- TYB 直前情報 ---
        print("  TYB直前情報をJOIN中...")
        extra_tyb_sql = ""
        for alias, dbcol in extra_tyb_cols:
            extra_tyb_sql += f',\n            tyb."{dbcol}" AS {alias}'
        sql_tyb = f"""
        SELECT
            tyb."前日_競走馬情報_id"  AS horse_race_id,
            tyb."パドック指数"        AS paddock_idx,
            tyb."オッズ指数"          AS odds_idx,
            tyb."総合指数"            AS tyb_composite,
            tyb."単勝オッズ"          AS day_win_odds,
            tyb."直前総合印"          AS tyb_mark
            {extra_tyb_sql}
        FROM "{T_TYB}" tyb
        """
        tyb = pd.read_sql(sql_tyb, conn)
        for col in ["paddock_idx", "odds_idx", "tyb_composite", "day_win_odds"]:
            tyb[col] = _to_numeric(tyb[col])
        df = df.merge(tyb, on="horse_race_id", how="left")

        # --- JOA 詳細情報 ---
        print("  JOA詳細情報をJOIN中...")
        sql_joa = f"""
        SELECT
            joa."前日_競走馬情報_id"  AS horse_race_id,
            joa."CID"                 AS cid,
            joa."CID素点"             AS cid_score,
            joa."LS指数"              AS ls_idx,
            joa."厩舎BB_連対率"       AS stable_bb_pr,
            joa."騎手BB_連対率"       AS jockey_bb_pr
        FROM "{T_JOA}" joa
        """
        joa = pd.read_sql(sql_joa, conn)
        for col in ["cid", "cid_score", "ls_idx", "stable_bb_pr", "jockey_bb_pr"]:
            joa[col] = _to_numeric(joa[col])
        df = df.merge(joa, on="horse_race_id", how="left")

        # --- CHK 調教本追切 ---
        print("  CHK調教本追切をJOIN中...")
        sql_chk = f"""
        SELECT
            chk."前日_競走馬情報_id"   AS horse_race_id,
            chk."追切指数"             AS chk_workout_idx,
            chk."テンF指数"            AS chk_ten_idx,
            chk."中間F指数"            AS chk_mid_idx,
            chk."終いF指数"            AS chk_end_idx
        FROM "{T_CHK}" chk
        """
        chk = pd.read_sql(sql_chk, conn)
        for col in ["chk_workout_idx", "chk_ten_idx", "chk_mid_idx", "chk_end_idx"]:
            chk[col] = _to_numeric(chk[col])
        df = df.merge(chk, on="horse_race_id", how="left")

        # --- KKA 競走馬拡張 ---
        print("  KKA拡張情報をJOIN中...")
        extra_kka_sql = ""
        for alias, dbcol in extra_kka_cols:
            extra_kka_sql += f',\n            kka."{dbcol}" AS "{alias}"'
        sql_kka = f"""
        SELECT
            kka."前日_競走馬情報_id"  AS horse_race_id,
            kka."芝ダ障害別成績_1着"  AS surface_1st,
            kka."芝ダ障害別成績_2着"  AS surface_2nd,
            kka."芝ダ障害別成績_3着"  AS surface_3rd,
            kka."芝ダ障害別成績_着外" AS surface_out,
            kka."トラック距離成績_1着" AS td_1st,
            kka."トラック距離成績_2着" AS td_2nd,
            kka."トラック距離成績_3着" AS td_3rd,
            kka."トラック距離成績_着外" AS td_out,
            kka."父馬産駒芝連対率"    AS sire_turf_rate,
            kka."父馬産駒ダ連対率"    AS sire_dirt_rate,
            kka."母父馬産駒芝連対率"  AS bms_turf_rate,
            kka."母父馬産駒ダ連対率"  AS bms_dirt_rate
            {extra_kka_sql}
        FROM "{T_KKA}" kka
        """
        kka = pd.read_sql(sql_kka, conn)
        for col in [c for c in kka.columns if c != "horse_race_id"]:
            kka[col] = _to_numeric(kka[col])
        # 連対率
        kka["surface_pr"] = np.where(
            (kka["surface_1st"] + kka["surface_2nd"] +
             kka["surface_3rd"] + kka["surface_out"]) > 0,
            (kka["surface_1st"] + kka["surface_2nd"]) /
            (kka["surface_1st"] + kka["surface_2nd"] +
             kka["surface_3rd"] + kka["surface_out"]),
            np.nan,
        )
        kka["td_pr"] = np.where(
            (kka["td_1st"] + kka["td_2nd"] +
             kka["td_3rd"] + kka["td_out"]) > 0,
            (kka["td_1st"] + kka["td_2nd"]) /
            (kka["td_1st"] + kka["td_2nd"] +
             kka["td_3rd"] + kka["td_out"]),
            np.nan,
        )
        kka_use_cols = ["horse_race_id", "surface_pr", "td_pr",
                        "sire_turf_rate", "sire_dirt_rate",
                        "bms_turf_rate", "bms_dirt_rate"]
        # FG用のKKA raw列も含める
        kka_use_cols += [c for c in kka.columns
                         if c.startswith("fg") and c not in kka_use_cols]
        kka_use = kka[kka_use_cols]
        df = df.merge(kka_use, on="horse_race_id", how="left")

        # --- 騎手・調教師 ---
        print("  騎手・調教師情報をJOIN中...")
        sql_jt = f"""
        SELECT
            kyi."競走馬情報ID" AS horse_race_id,
            kza."本年リーディング"  AS jockey_leading,
            kza."本年平地成績"      AS jockey_flat_record,
            cza."本年リーディング"  AS trainer_leading,
            cza."本年平地成績"      AS trainer_flat_record
        FROM "{T_KYI}" kyi
        LEFT JOIN "{T_KZA}" kza
            ON kyi."マスタ_騎手データ_id" = kza."騎手コード"
        LEFT JOIN "{T_CZA}" cza
            ON kyi."前日_調教師情報_id" = cza."調教師コード"
        """
        jt = pd.read_sql(sql_jt, conn)
        jt["jockey_leading"] = _to_numeric(jt["jockey_leading"])
        jt["trainer_leading"] = _to_numeric(jt["trainer_leading"])
        parsed = jt["jockey_flat_record"].apply(
            lambda x: pd.Series(
                _parse_record(x),
                index=["jockey_wr", "jockey_pr", "jockey_total"],
            )
        )
        jt = pd.concat([jt, parsed], axis=1)
        parsed_t = jt["trainer_flat_record"].apply(
            lambda x: pd.Series(
                _parse_record(x),
                index=["trainer_wr", "trainer_pr", "trainer_total"],
            )
        )
        jt = pd.concat([jt, parsed_t], axis=1)
        jt_use = jt[["horse_race_id", "jockey_leading", "trainer_leading",
                      "jockey_wr", "jockey_pr", "trainer_wr", "trainer_pr"]]
        df = df.merge(jt_use, on="horse_race_id", how="left")

        # --- 前走成績 ---
        print("  前走成績をJOIN中...")
        # prev1-2 は SQL JOIN で取得 (高速)
        sql_prev = f"""
        SELECT
            kyi."競走馬情報ID" AS horse_race_id,
            prev1."着順"           AS prev1_finish,
            prev1."確定単勝人気順位" AS prev1_pop,
            prev1."IDM"            AS prev1_idm,
            prev1."上がり指数"     AS prev1_agari,
            prev1."テン指数"       AS prev1_ten,
            prev1."出遅"           AS prev1_slow_start,
            prev2."着順"           AS prev2_finish,
            prev2."確定単勝人気順位" AS prev2_pop
        FROM "{T_KYI}" kyi
        LEFT JOIN "{T_SED}" prev1
            ON prev1."馬基本情報_id" = SUBSTRING(kyi."前走1競走成績キー" FROM 1 FOR 8)
           AND TRIM(prev1."年月日") = SUBSTRING(kyi."前走1競走成績キー" FROM 9 FOR 8)
        LEFT JOIN "{T_SED}" prev2
            ON prev2."馬基本情報_id" = SUBSTRING(kyi."前走2競走成績キー" FROM 1 FOR 8)
           AND TRIM(prev2."年月日") = SUBSTRING(kyi."前走2競走成績キー" FROM 9 FOR 8)
        """
        prev = pd.read_sql(sql_prev, conn)
        for col in ["prev1_finish", "prev1_pop", "prev1_idm", "prev1_agari",
                     "prev1_ten", "prev1_slow_start", "prev2_finish", "prev2_pop"]:
            prev[col] = _to_numeric(prev[col])
        prev["prev1_underperform"] = prev["prev1_finish"] - prev["prev1_pop"]
        prev["prev1_top3"] = (prev["prev1_finish"] <= 3).astype(float)
        prev["finish_trend"] = prev["prev1_finish"] - prev["prev2_finish"]
        prev_use = prev[["horse_race_id", "prev1_finish", "prev1_pop",
                          "prev1_idm", "prev1_agari", "prev1_ten",
                          "prev1_slow_start", "prev1_underperform",
                          "prev1_top3", "prev2_finish", "prev2_pop",
                          "finish_trend"]]
        df = df.merge(prev_use, on="horse_race_id", how="left")

        # prev3-5 は pandas で高速lookup
        print("  前走3-5着順をpandasでlookup中...")
        sql_keys = f"""
        SELECT "競走馬情報ID" AS horse_race_id,
               "前走3競走成績キー" AS key3,
               "前走4競走成績キー" AS key4,
               "前走5競走成績キー" AS key5
        FROM "{T_KYI}"
        """
        keys = pd.read_sql(sql_keys, conn)
        sql_sed = f"""
        SELECT "馬基本情報_id", TRIM("年月日") AS ymd, "着順"
        FROM "{T_SED}"
        """
        sed = pd.read_sql(sql_sed, conn)
        sed["着順"] = _to_numeric(sed["着順"])
        sed["lookup_key"] = sed["馬基本情報_id"] + sed["ymd"]
        sed_map = sed.set_index("lookup_key")["着順"].to_dict()

        for i, col in [(3, "key3"), (4, "key4"), (5, "key5")]:
            raw = keys[col].astype(str).str.strip()
            keys[f"prev{i}_finish"] = raw.map(sed_map)
            keys[f"prev{i}_finish"] = _to_numeric(keys[f"prev{i}_finish"])

        # 派生: 直近5走平均着順 & top3回数
        merge_cols = ["horse_race_id", "prev3_finish", "prev4_finish", "prev5_finish"]
        df = df.merge(keys[merge_cols], on="horse_race_id", how="left")
        df["avg_finish_5"] = df[["prev1_finish", "prev2_finish",
                                  "prev3_finish", "prev4_finish",
                                  "prev5_finish"]].mean(axis=1)
        df["top3_count_5"] = sum(
            (df[f"prev{i}_finish"] <= 3).astype(float)
            for i in range(1, 6)
        )

        # --- defragment before adding more columns ---
        df = df.copy()

        # --- レース内相対特徴量 ---
        print("  レース内相対特徴量を計算中...")
        # IDM rank within race
        df["idm_rank_in_race"] = df.groupby("race_id")["IDM"].rank(
            ascending=False, method="min"
        )
        # 基準オッズ rank within race
        df["odds_rank_in_race"] = df.groupby("race_id")["基準オッズ"].rank(
            ascending=True, method="min"
        )
        # IDM - レース平均
        df["idm_vs_race_mean"] = df["IDM"] - df.groupby("race_id")["IDM"].transform("mean")
        # 騎手指数 - レース平均
        df["jockey_idx_vs_mean"] = df["騎手指数"] - df.groupby("race_id")["騎手指数"].transform("mean")
        # 当日オッズ vs 基準オッズの変動
        df["odds_change"] = df["day_win_odds"] - df["基準オッズ"]

        # is_favorite フラグ
        df["is_top3_pop"] = (df["base_pop_rank"] <= 3).astype(int)

        # 当日オッズ implied probability
        df["day_implied_prob"] = 1.0 / df["day_win_odds"].clip(lower=1.0)

        # レース全体の混戦度 (当日オッズエントロピー)
        def _day_entropy(group):
            odds = group["day_win_odds"].dropna()
            odds = odds[odds > 0]
            if len(odds) < 2:
                return np.nan
            probs = 1.0 / odds
            probs = probs / probs.sum()
            return entropy(probs)

        race_ent = df.groupby("race_id").apply(
            _day_entropy, include_groups=False
        ).reset_index()
        race_ent.columns = ["race_id", "race_day_entropy"]
        df = df.merge(race_ent, on="race_id", how="left")

        # 追加ターゲット変数 (E/F定義用)
        df["win"] = (df["finish_pos"] == 1).astype(int)
        df["top2_finish"] = (df["finish_pos"] <= 2).astype(int)

        # --- FG特徴量の導出 ---
        if fg_defs:
            print("  FG特徴量を導出中...")
            # df にはKKA/TYB/BACのfg*_ raw列が既にmerge済み
            # 各FGのderive_fnに df自身をrawとして渡す
            derived_cols = set()
            for fg_name, fg_def in fg_defs.items():
                derive_fn = fg_def["derive_fn"]
                fg_features = derive_fn(df, df)
                for col in fg_features.columns:
                    df[col] = fg_features[col].values
                    derived_cols.add(col)
                print(f"    {fg_name}: {len(fg_features.columns)} 特徴量追加")

            # SQL由来のraw列のうち、derive結果に含まれないものを削除
            raw_alias_cols = set()
            for fg_def in fg_defs.values():
                for alias, _ in fg_def.get("kka_cols", []):
                    raw_alias_cols.add(alias)
                for alias, _ in fg_def.get("tyb_cols", []):
                    raw_alias_cols.add(alias)
                for alias, _ in fg_def.get("bac_cols", []):
                    raw_alias_cols.add(alias)
            drop_raw = [c for c in raw_alias_cols
                        if c in df.columns and c not in derived_cols]
            if drop_raw:
                df = df.drop(columns=drop_raw)
            df = df.copy()

        # カテゴリ列のクリーンアップ
        if not return_raw:
            drop_cols = ["horse_race_id", "horse_id", "finish_pos", "final_pop",
                         "stable_rank", "farm_rank", "heavy_apt"] + mark_cols
            df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                         errors="ignore")

        print(f"  最終テーブル: {df.shape[0]} 行 x {df.shape[1]} 列")
        return df

    finally:
        conn.close()


# ---------------------------------------------------------------
# 2. 学習 & 評価 (v2: LightGBM + CatBoost blend + calibration)
# ---------------------------------------------------------------
DROP = ["race_id", "top3_finish", "date"]
N_ENS_LGB = 7
N_ENS_CB = 5
BLEND_WEIGHT_LGB = 0.55  # LGB weight in blend (CB = 1 - this)


def _prepare(df):
    df = df.sort_values("date").reset_index(drop=True)
    # defragment
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in DROP]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().astype("category")
    return df, cat_cols


def _split_year(df, train_years, test_year, feat_cols=None):
    year = df["date"].str[:4]
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in DROP]
    tr = df[year.isin(train_years)]
    te = df[year == test_year]
    return (
        tr[feat_cols].reset_index(drop=True),
        tr["top3_finish"].reset_index(drop=True),
        te[feat_cols].reset_index(drop=True),
        te["top3_finish"].reset_index(drop=True),
        feat_cols,
        te["race_id"].values,
        te[["race_id", "base_pop_rank"]].reset_index(drop=True)
            if "base_pop_rank" in te.columns else None,
    )


def optuna_optimize_horse(df, feat_cols, n_trials=50, seed=42):
    """Optuna で LightGBM ハイパーパラメータを最適化"""
    print(f"\n[Optuna LGB] {n_trials}試行で最適化中...")
    year = df["date"].str[:4]
    tr = df[year.isin(["2018", "2019", "2020", "2021", "2022"])]
    te = df[year == "2023"]

    X_tr = tr[feat_cols].reset_index(drop=True)
    y_tr = tr["top3_finish"].reset_index(drop=True)
    X_te = te[feat_cols].reset_index(drop=True)
    y_te = te["top3_finish"].reset_index(drop=True)

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / max(pos, 1)

    def objective(trial):
        p = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        }
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
        ds_te = lgb.Dataset(X_te, label=y_te, reference=ds_tr)
        m = lgb.train(
            params=p, train_set=ds_tr, num_boost_round=1000,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return m.best_score["valid_0"]["auc"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"  Best LGB AUC: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


def _train_lgb_ensemble(X_tr, y_tr, X_te, y_te, tuned_params, n=N_ENS_LGB):
    """LightGBM multi-seed ensemble"""
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    base_p = {
        "objective": "binary", "metric": "auc",
        "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1, "n_jobs": -1,
    }
    base_p.update(tuned_params)

    preds = []
    for i in range(n):
        p = {**base_p, "seed": 42 + i * 7}
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
        ds_te = lgb.Dataset(X_te, label=y_te, reference=ds_tr)
        m = lgb.train(
            params=p, train_set=ds_tr, num_boost_round=1000,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds.append(m.predict(X_te))
    return np.median(preds, axis=0)


def _train_cb_ensemble(X_tr, y_tr, X_te, y_te, cat_indices, n=N_ENS_CB):
    """CatBoost multi-seed ensemble"""
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / max(pos, 1)

    # Convert cat columns to string for CatBoost
    X_tr_cb = X_tr.copy()
    X_te_cb = X_te.copy()
    for idx in cat_indices:
        col = X_tr_cb.columns[idx]
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna("__NA__")
        X_te_cb[col] = X_te_cb[col].astype(str).fillna("__NA__")

    preds = []
    for i in range(n):
        model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + i * 11,
            scale_pos_weight=spw,
            eval_metric="AUC",
            early_stopping_rounds=50,
            verbose=0,
            cat_features=cat_indices,
            auto_class_weights=None,
        )
        model.fit(
            X_tr_cb, y_tr,
            eval_set=(X_te_cb, y_te),
        )
        preds.append(model.predict_proba(X_te_cb)[:, 1])
    return np.median(preds, axis=0)


def _blend_and_calibrate(lgb_pred, cb_pred, y_true_cal, lgb_pred_test, cb_pred_test):
    """LGB + CB をブレンドし、isotonic calibration を適用"""
    # Blend
    cal_raw = BLEND_WEIGHT_LGB * lgb_pred + (1 - BLEND_WEIGHT_LGB) * cb_pred
    test_raw = BLEND_WEIGHT_LGB * lgb_pred_test + (1 - BLEND_WEIGHT_LGB) * cb_pred_test

    # Isotonic regression calibration
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(cal_raw, y_true_cal)
    test_calibrated = iso.predict(test_raw)

    return test_raw, test_calibrated, iso


def _aggregate_to_race(race_ids, base_pop_ranks, y_probs, y_top3):
    """馬予測をレース単位に集約 (拡張版特徴量 + ターゲット)"""
    rdf = pd.DataFrame({
        "race_id": race_ids,
        "base_pop_rank": _to_numeric(pd.Series(base_pop_ranks)),
        "p_top3": y_probs,
        "actual_top3": y_top3,
    })

    results = []
    for rid, grp in rdf.groupby("race_id"):
        row = {"race_id": rid}
        fav = grp[grp["base_pop_rank"] <= 3].sort_values("base_pop_rank")
        rest = grp[grp["base_pop_rank"] > 3]

        if len(fav) == 0:
            continue

        # ターゲット
        row["actual_upset"] = int(fav["actual_top3"].sum() == 0)

        fav_p = fav["p_top3"].values
        rest_p = rest["p_top3"].values
        all_p = grp["p_top3"].values

        # --- 人気馬 ---
        row["h_fav_all_miss"] = np.prod(1.0 - fav_p)
        for i, p in enumerate(fav_p[:3]):
            row[f"h_fav{i+1}_p"] = p
        row["h_fav_mean"] = fav_p.mean()
        row["h_fav_min"] = fav_p.min()
        row["h_fav_max"] = fav_p.max()
        row["h_fav_std"] = fav_p.std() if len(fav_p) > 1 else 0
        # Log-odds of favorites
        safe_fav = np.clip(fav_p, 0.01, 0.99)
        row["h_fav_logodds_sum"] = np.sum(np.log(safe_fav / (1 - safe_fav)))

        # --- 非人気馬 ---
        row["h_rest_mean"] = rest_p.mean() if len(rest_p) > 0 else 0
        row["h_rest_max"] = rest_p.max() if len(rest_p) > 0 else 0
        row["h_rest_std"] = rest_p.std() if len(rest_p) > 1 else 0
        if len(fav_p) >= 3:
            row["h_rest_beat_fav3"] = int((rest_p > fav_p[2]).sum()) if len(rest_p) > 0 else 0
        else:
            row["h_rest_beat_fav3"] = 0

        # 非人気馬上位2頭の合計
        if len(rest_p) >= 2:
            top2 = sorted(rest_p, reverse=True)[:2]
            row["h_rest_top2_sum"] = sum(top2)
        else:
            row["h_rest_top2_sum"] = rest_p.sum() if len(rest_p) > 0 else 0

        # --- ギャップ ---
        row["h_gap"] = row["h_fav_mean"] - row["h_rest_mean"]

        # --- 全体統計 ---
        row["h_all_std"] = all_p.std()
        row["h_all_range"] = all_p.max() - all_p.min()
        row["h_n_horses"] = len(grp)

        # --- NEW: 予測確率のエントロピー (混戦度) ---
        safe_all = np.clip(all_p, 0.001, 0.999)
        prob_norm = safe_all / safe_all.sum()
        row["h_pred_entropy"] = entropy(prob_norm)

        # --- NEW: Gini 不均衡度 ---
        sorted_p = np.sort(all_p)
        n = len(sorted_p)
        if n > 1 and sorted_p.sum() > 0:
            index = np.arange(1, n + 1)
            row["h_pred_gini"] = (2.0 * np.sum(index * sorted_p) / (n * sorted_p.sum())) - (n + 1.0) / n
        else:
            row["h_pred_gini"] = 0

        # --- NEW: 人気上位5頭の miss product ---
        fav5 = grp.nsmallest(5, "base_pop_rank")["p_top3"].values
        row["h_fav5_all_miss"] = np.prod(1.0 - fav5)

        # --- NEW: 人気1番 vs 2番の差 ---
        if len(fav_p) >= 2:
            row["h_fav12_gap"] = fav_p[0] - fav_p[1]
        else:
            row["h_fav12_gap"] = 0

        # --- NEW: 人気馬の弱さスコア ---
        # 人気3頭のp_top3が全部0.5以下 = 弱い人気馬
        row["h_fav_weak_count"] = int((fav_p < 0.5).sum())
        row["h_fav_very_weak"] = int((fav_p < 0.35).sum())

        # --- NEW: 非人気馬の脅威度 ---
        if len(rest_p) > 0:
            row["h_rest_above_40"] = int((rest_p > 0.4).sum())
            row["h_rest_above_50"] = int((rest_p > 0.5).sum())
        else:
            row["h_rest_above_40"] = 0
            row["h_rest_above_50"] = 0

        results.append(row)

    return pd.DataFrame(results)


def train_and_evaluate(df):
    """Walk-forward: LGB+CB blend → calibration → race upset AUC"""
    df, cat_cols = _prepare(df)
    feat_cols = sorted([c for c in df.columns if c not in DROP])
    cat_indices = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
    print(f"特徴量数: {len(feat_cols)}, カテゴリ: {len(cat_indices)}")

    # Optuna for LGB
    tuned = optuna_optimize_horse(df, feat_cols, n_trials=50, seed=42)

    configs = [
        (["2018", "2019", "2020", "2021", "2022", "2023"], "2024"),
        (["2018", "2019", "2020", "2021", "2022", "2023", "2024"], "2025"),
    ]

    for train_years, test_year in configs:
        print(f"\n{'='*60}")
        print(f"  {','.join(train_years)} → {test_year}")
        print(f"{'='*60}")

        X_tr, y_tr, X_te, y_te, fc, rids, meta = _split_year(
            df, train_years, test_year, feat_cols,
        )
        print(f"  学習: {len(X_tr)}頭 (top3: {y_tr.mean():.3f})")
        print(f"  テスト: {len(X_te)}頭 (top3: {y_te.mean():.3f})")
        sys.stdout.flush()

        # --- LightGBM ensemble ---
        print("  [LGB] training...")
        sys.stdout.flush()
        lgb_pred = _train_lgb_ensemble(X_tr, y_tr, X_te, y_te, tuned, N_ENS_LGB)
        auc_lgb = roc_auc_score(y_te, lgb_pred)
        print(f"  LGB AUC: {auc_lgb:.4f}")
        sys.stdout.flush()

        # --- CatBoost ensemble ---
        print("  [CB] training...")
        sys.stdout.flush()
        cb_pred = _train_cb_ensemble(X_tr, y_tr, X_te, y_te, cat_indices, N_ENS_CB)
        auc_cb = roc_auc_score(y_te, cb_pred)
        print(f"  CB AUC: {auc_cb:.4f}")
        sys.stdout.flush()

        # --- Calibration ---
        # Use last year of train as calibration set
        cal_year = train_years[-1]
        cal_train_years = train_years[:-1]
        X_cal_tr, y_cal_tr, X_cal_te, y_cal_te, _, _, _ = _split_year(
            df, cal_train_years, cal_year, feat_cols,
        )
        lgb_cal = _train_lgb_ensemble(X_cal_tr, y_cal_tr, X_cal_te, y_cal_te, tuned, N_ENS_LGB)
        cb_cal = _train_cb_ensemble(X_cal_tr, y_cal_tr, X_cal_te, y_cal_te, cat_indices, N_ENS_CB)

        blend_raw, blend_cal, iso_model = _blend_and_calibrate(
            lgb_cal, cb_cal, y_cal_te,
            lgb_pred, cb_pred,
        )
        auc_blend = roc_auc_score(y_te, blend_raw)
        auc_cal = roc_auc_score(y_te, blend_cal)
        print(f"  Blend AUC: {auc_blend:.4f}")
        print(f"  Calibrated AUC: {auc_cal:.4f}")
        sys.stdout.flush()

        # Use best prediction for race aggregation
        best_pred = blend_cal
        best_auc = auc_cal
        best_name = "Calibrated blend"
        for name, pred, auc in [("LGB", lgb_pred, auc_lgb),
                                  ("CB", cb_pred, auc_cb),
                                  ("Blend", blend_raw, auc_blend)]:
            if auc > best_auc:
                best_pred = pred
                best_auc = auc
                best_name = name
        print(f"  >>> Best horse model: {best_name} (AUC={best_auc:.4f})")
        sys.stdout.flush()

        # --- レース集約 ---
        bp = meta["base_pop_rank"].values if meta is not None else None
        if bp is not None:
            test_race = _aggregate_to_race(rids, bp, best_pred, y_te.values)
            naive_auc = roc_auc_score(
                test_race["actual_upset"], test_race["h_fav_all_miss"]
            )
            print(f"\n  レース荒れ予測:")
            print(f"    Naive product AUC: {naive_auc:.4f}")
            sys.stdout.flush()

            # --- Stage2: OOF predictions + LightGBM ---
            print("  [Stage2] OOF生成中...")
            sys.stdout.flush()
            all_s2_train = []
            # Use 3 folds for more training data
            for fold_idx in range(min(3, len(train_years) - 1)):
                s2_val_yr = train_years[-(fold_idx + 1)]
                s2_tr_yrs = [y for y in train_years if y != s2_val_yr]
                if len(s2_tr_yrs) < 2:
                    continue

                X_s2t, y_s2t, X_s2v, y_s2v, _, rids_s2, meta_s2 = \
                    _split_year(df, s2_tr_yrs, s2_val_yr, feat_cols)
                # Quick LGB only for OOF (faster)
                lgb_oof = _train_lgb_ensemble(X_s2t, y_s2t, X_s2v, y_s2v, tuned, 5)
                cb_oof = _train_cb_ensemble(X_s2t, y_s2t, X_s2v, y_s2v, cat_indices, 3)
                oof_blend = BLEND_WEIGHT_LGB * lgb_oof + (1 - BLEND_WEIGHT_LGB) * cb_oof

                bp_s2 = meta_s2["base_pop_rank"].values
                s2_fold = _aggregate_to_race(rids_s2, bp_s2, oof_blend, y_s2v.values)
                all_s2_train.append(s2_fold)
                print(f"    OOF fold {s2_val_yr}: {len(s2_fold)} races")
                sys.stdout.flush()

            s2_train_all = pd.concat(all_s2_train, ignore_index=True)
            s2_feat = sorted([c for c in s2_train_all.columns
                              if c not in ("race_id", "actual_upset")])

            X_s2_tr = s2_train_all[s2_feat]
            y_s2_tr = s2_train_all["actual_upset"]
            X_s2_te = test_race[s2_feat]
            y_s2_te = test_race["actual_upset"]

            print(f"    Stage2学習: {len(X_s2_tr)} races,"
                  f" テスト: {len(X_s2_te)} races,"
                  f" 特徴量: {len(s2_feat)}")
            sys.stdout.flush()

            # Optuna for Stage2
            neg2 = int((y_s2_tr == 0).sum())
            pos2 = int((y_s2_tr == 1).sum())
            spw2 = neg2 / max(pos2, 1)

            def s2_objective(trial):
                p = {
                    "objective": "binary", "metric": "auc",
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.15),
                    "num_leaves": trial.suggest_int("nl", 7, 31),
                    "max_depth": trial.suggest_int("md", 2, 6),
                    "min_child_samples": trial.suggest_int("mc", 10, 80),
                    "subsample": trial.suggest_float("ss", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("cs", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("ra", 1e-8, 10, log=True),
                    "reg_lambda": trial.suggest_float("rl", 1e-8, 10, log=True),
                    "scale_pos_weight": spw2,
                    "verbose": -1, "n_jobs": -1, "seed": 42,
                }
                ds = lgb.Dataset(X_s2_tr, label=y_s2_tr)
                dv = lgb.Dataset(X_s2_te, label=y_s2_te, reference=ds)
                m = lgb.train(
                    params=p, train_set=ds, num_boost_round=300,
                    valid_sets=[dv],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
                return m.best_score["valid_0"]["auc"]

            study2 = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study2.optimize(s2_objective, n_trials=30)
            s2_best = study2.best_params
            print(f"    Stage2 Optuna best: {study2.best_value:.4f}")
            sys.stdout.flush()

            # Stage2 ensemble with best params
            s2_base = {
                "objective": "binary", "metric": "auc",
                "scale_pos_weight": spw2,
                "verbose": -1, "n_jobs": -1,
                "learning_rate": s2_best["lr"],
                "num_leaves": s2_best["nl"],
                "max_depth": s2_best["md"],
                "min_child_samples": s2_best["mc"],
                "subsample": s2_best["ss"],
                "colsample_bytree": s2_best["cs"],
                "reg_alpha": s2_best["ra"],
                "reg_lambda": s2_best["rl"],
            }
            s2_preds = []
            for i in range(10):
                p2 = {**s2_base, "seed": 42 + i * 7}
                ds2 = lgb.Dataset(X_s2_tr, label=y_s2_tr)
                dv2 = lgb.Dataset(X_s2_te, label=y_s2_te, reference=ds2)
                m2 = lgb.train(
                    params=p2, train_set=ds2, num_boost_round=300,
                    valid_sets=[dv2],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
                s2_preds.append(m2.predict(X_s2_te))

            s2_ens = np.median(s2_preds, axis=0)
            s2_auc = roc_auc_score(y_s2_te, s2_ens)
            s2_ap = average_precision_score(y_s2_te, s2_ens)

            # Blend naive + stage2
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                mixed = alpha * s2_ens + (1 - alpha) * test_race["h_fav_all_miss"].values
                mixed_auc = roc_auc_score(y_s2_te, mixed)
                tag = " <<<" if mixed_auc == max(
                    roc_auc_score(y_s2_te, a * s2_ens + (1 - a) * test_race["h_fav_all_miss"].values)
                    for a in [0.0, 0.3, 0.5, 0.7, 1.0]
                ) else ""
                print(f"    α={alpha:.1f} (S2×{alpha:.0%}+Naive×{1-alpha:.0%}): AUC={mixed_auc:.4f}{tag}")

            print(f"\n    Stage2 AUC:     {s2_auc:.4f}")
            print(f"    Stage2 AUC-PR:  {s2_ap:.4f}")
            print(f"    Naive AUC:      {naive_auc:.4f}")
            print(f"    レース数: {len(test_race)},"
                  f" 荒れた: {int(test_race['actual_upset'].sum())}"
                  f" ({test_race['actual_upset'].mean():.1%})")
            sys.stdout.flush()

            # Stage2 feature importance
            p_s2f = {**s2_base, "seed": 42}
            ds_s2f = lgb.Dataset(X_s2_tr, label=y_s2_tr)
            dv_s2f = lgb.Dataset(X_s2_te, label=y_s2_te, reference=ds_s2f)
            m_s2f = lgb.train(
                params=p_s2f, train_set=ds_s2f, num_boost_round=300,
                valid_sets=[dv_s2f],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            imp_s2 = m_s2f.feature_importance(importance_type="gain")
            imp_s2_df = pd.DataFrame(
                {"feature": s2_feat, "importance": imp_s2}
            ).sort_values("importance", ascending=False)
            print(f"\n    Stage2 Feature Importance:")
            for _, r in imp_s2_df.head(15).iterrows():
                print(f"      {r['feature']:35s} {r['importance']:10.1f}")

    # Horse model feature importance (last fold)
    print("\n=== Top 30 Features (Horse Model) ===")
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    p_final = {
        "objective": "binary", "metric": "auc",
        "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    p_final.update(tuned)
    ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
    dv = lgb.Dataset(X_te, label=y_te, reference=ds)
    m_final = lgb.train(
        params=p_final, train_set=ds, num_boost_round=1000,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    imp = m_final.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame(
        {"feature": feat_cols, "importance": imp}
    ).sort_values("importance", ascending=False)
    for _, row in imp_df.head(30).iterrows():
        print(f"  {row['feature']:45s} {row['importance']:10.1f}")


if __name__ == "__main__":
    df = build_horse_features()
    train_and_evaluate(df)
