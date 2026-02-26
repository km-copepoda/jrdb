"""
Top3-fav 荒れ予測 & ROI最適化 (E/F定義)

E: fav1,2,3が1着にこない → 単勝ex3
   馬モデル: P(win) = P(着順==1)
   p_product = (1-p_fav1)*(1-p_fav2)*(1-p_fav3)

F: fav1,2,3が1,2着にこない → 馬単ex3
   馬モデル: P(top2) = P(着順<=2)
   p_product = (1-p_fav1)*(1-p_fav2)*(1-p_fav3)

Rolling (7年スライディングウィンドウ):
  2015-2021→2022, 2016-2022→2023, 2017-2023→2024, 2018-2024→2025, 2019-2025→2026
"""
import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import psycopg2
import lightgbm as lgb
import catboost as cb
from math import comb, perm
from scipy.stats import entropy as _entropy
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegressionCV
from horse_model import build_horse_features, _to_numeric, _prepare
from config import DB_CONFIG, T_BAC, T_HJC, T_SED, T_KYI, T_TYB
from evaluate_factors import _load_extra_prev_cols, compute_factors
from factor_definitions import FACTOR_CATALOG, ALL_FACTOR_NAMES

# 最適特徴量グループ (実験結果: FG2+FG3+FG4+FG5+FG6)
# FG2: ペース/季節/枠, FG3: 騎手専門性, FG4: 馬体重,
# FG5: 重量種別+ローテ, FG6: コンボ+レース集約
BEST_FEATURE_GROUPS = ["FG2", "FG3", "FG4", "FG5", "FG6"]

warnings.filterwarnings("ignore")

# 全ターゲット列 + ID列 (feat_colsから除外)
DROP = ["race_id", "top3_finish", "win", "top2_finish", "date"]

N_ENS_CB = 3
BLEND_WEIGHT_LGB = 0.55

# L1-onlyモード: LGB/CBをスキップし、オッズ由来確率+factor_scoreのみ使用
L1_ONLY = False

# return_raw=True で残るが ML特徴量としては不要な列
_RAW_DROP = [
    "horse_race_id", "horse_id", "finish_pos", "final_pop",
    "stable_rank", "farm_rank", "heavy_apt", "horse_no",
    "mark_overall", "mark_idm", "mark_info", "mark_jockey",
    "mark_stable", "mark_training", "mark_upset", "mark_longshot",
    "prev1_furi", "prev1_mae_furi", "prev1_naka_furi",
    "prev1_ato_furi", "prev1_race_eval", "prev1_soten",
    "prev1_mae3f", "prev1_ato3f", "prev1_kinryo",
    "prev1_distance", "prev1_surface", "prev1_grade",
    "prev1_weight", "prev1_weight_change", "prev2_furi",
]

def _fit_l1_weights(factor_binary_store, factor_cols, l1_yrs):
    """L1正則化ロジスティック回帰でファクター重みをフォールド内推定"""
    mask = factor_binary_store["date"].str[:4].isin(l1_yrs)
    l1_df = factor_binary_store[mask]
    y = l1_df["upset_horse"].values
    X = l1_df[factor_cols].fillna(0).astype(float)

    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    if len(y) < 100 or y.sum() < 10:
        return {}

    model = LogisticRegressionCV(
        Cs=20, cv=5, penalty='l1', solver='saga',
        scoring='roc_auc', max_iter=5000, random_state=42,
    )
    model.fit(X, y)
    return {fc: w for fc, w in zip(factor_cols, model.coef_[0])
            if abs(w) > 1e-6}


TUNED_PARAMS = {
    "learning_rate": 0.078,
    "num_leaves": 92,
    "max_depth": 3,
    "min_child_samples": 34,
    "subsample": 0.648,
    "colsample_bytree": 0.581,
    "reg_alpha": 1.01e-07,
    "reg_lambda": 9.09e-06,
}


# ---------------------------------------------------------------
# 馬モデル学習
# ---------------------------------------------------------------
def _train_lgb(X_tr, y_tr, X_te, y_te, n=7):
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    base_p = {
        "objective": "binary", "metric": "auc",
        "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1, "n_jobs": -1,
    }
    base_p.update(TUNED_PARAMS)
    preds = []
    for i in range(n):
        p = {**base_p, "seed": 42 + i * 7}
        ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
        dv = lgb.Dataset(X_te, label=y_te, reference=ds)
        m = lgb.train(
            params=p, train_set=ds, num_boost_round=1200,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds.append(m.predict(X_te))
    return np.median(preds, axis=0)


def _train_cb(X_tr, y_tr, X_te, y_te, cat_indices, n=N_ENS_CB):
    """CatBoost multi-seed ensemble"""
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / max(pos, 1)
    X_tr_cb = X_tr.copy()
    X_te_cb = X_te.copy()
    for idx in cat_indices:
        col = X_tr_cb.columns[idx]
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna("__NA__")
        X_te_cb[col] = X_te_cb[col].astype(str).fillna("__NA__")
    preds = []
    for i in range(n):
        model = cb.CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, random_seed=42 + i * 11,
            scale_pos_weight=spw, eval_metric="AUC",
            early_stopping_rounds=50, verbose=0,
            cat_features=cat_indices,
        )
        model.fit(X_tr_cb, y_tr, eval_set=(X_te_cb, y_te))
        preds.append(model.predict_proba(X_te_cb)[:, 1])
    return np.median(preds, axis=0)


def _blend_and_calibrate(lgb_cal, cb_cal, y_cal,
                         lgb_test, cb_test):
    """LGB+CBブレンド → isotonic calibration"""
    cal_raw = BLEND_WEIGHT_LGB * lgb_cal + (1 - BLEND_WEIGHT_LGB) * cb_cal
    test_raw = BLEND_WEIGHT_LGB * lgb_test + (1 - BLEND_WEIGHT_LGB) * cb_test
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)
    return iso.predict(test_raw)


# ---------------------------------------------------------------
# DB取得
# ---------------------------------------------------------------
def _get_final_pop(race_ids):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ids_sql = ", ".join(f"'{r}'" for r in race_ids)
        sql = f"""
        SELECT sed."前日_番組情報_id" AS race_id,
               sed."確定単勝人気順位" AS final_pop,
               sed."着順" AS finish_pos
        FROM "{T_SED}" sed
        WHERE sed."前日_番組情報_id" IN ({ids_sql})
        """
        df = pd.read_sql(sql, conn)
        df["final_pop"] = _to_numeric(df["final_pop"])
        df["finish_pos"] = _to_numeric(df["finish_pos"])
        return df
    finally:
        conn.close()


def _get_hjc_full(race_ids):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ids_sql = ", ".join(f"'{r}'" for r in race_ids)
        cols = []
        for prefix in ["単勝払戻1", "単勝払戻2", "単勝払戻3"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番"]
        for prefix in ["複勝払戻1", "複勝払戻2", "複勝払戻3",
                       "複勝払戻4", "複勝払戻5"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番"]
        for prefix in ["馬連払戻1", "馬連払戻2", "馬連払戻3"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番組合せ"]
        for prefix in ["馬単払戻1", "馬単払戻2", "馬単払戻3",
                       "馬単払戻4", "馬単払戻5", "馬単払戻6"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番組合せ"]
        for prefix in ["三連複払戻1", "三連複払戻2", "三連複払戻3"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番組合せ"]
        for prefix in ["三連単払戻1", "三連単払戻2", "三連単払戻3",
                       "三連単払戻4", "三連単払戻5", "三連単払戻6"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番組合せ"]
        for prefix in ["ワイド払戻1", "ワイド払戻2", "ワイド払戻3",
                       "ワイド払戻4", "ワイド払戻5", "ワイド払戻6",
                       "ワイド払戻7"]:
            cols += [f"{prefix}_払戻金", f"{prefix}_馬番組合せ"]

        cols_sql = ", ".join(f'hjc."{c}"' for c in cols)
        sql = f"""
        SELECT hjc."前日_番組情報_id" AS race_id, {cols_sql}
        FROM "{T_HJC}" hjc
        WHERE hjc."前日_番組情報_id" IN ({ids_sql})
        """
        df = pd.read_sql(sql, conn)
        for c in [c for c in cols if "払戻金" in c]:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.strip(), errors="coerce"
            ).fillna(0)
        for c in [c for c in cols if "組合せ" in c or "馬番" in c]:
            df[c] = df[c].astype(str).str.strip()
        return df
    finally:
        conn.close()


def _get_n_horses(race_ids):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ids_sql = ", ".join(f"'{r}'" for r in race_ids)
        sql = f"""
        SELECT bac."番組情報ID" AS race_id, bac."頭数" AS n_horses
        FROM "{T_BAC}" bac
        WHERE bac."番組情報ID" IN ({ids_sql})
        """
        df = pd.read_sql(sql, conn)
        df["n_horses"] = pd.to_numeric(
            df["n_horses"].astype(str).str.strip(), errors="coerce"
        ).fillna(14).astype(int)
        return df.set_index("race_id")["n_horses"].to_dict()
    finally:
        conn.close()


def _get_fav_umabans_top3(race_ids):
    """KYI+TYB: レース毎のfav1/fav2/fav3の馬番を取得"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ids_sql = ", ".join(f"'{r}'" for r in race_ids)
        sql = f"""
        SELECT kyi."前日_番組情報_id" AS race_id,
               kyi."馬番" AS umaban,
               tyb."単勝オッズ" AS day_odds
        FROM "{T_KYI}" kyi
        LEFT JOIN "{T_TYB}" tyb
            ON tyb."前日_競走馬情報_id" = kyi."競走馬情報ID"
        WHERE kyi."前日_番組情報_id" IN ({ids_sql})
        """
        df = pd.read_sql(sql, conn)
        df["umaban"] = df["umaban"].astype(str).str.strip().str.zfill(2)
        df["day_odds"] = pd.to_numeric(
            df["day_odds"].astype(str).str.strip(), errors="coerce"
        )
        result = {}
        for rid, grp in df.groupby("race_id"):
            grp_v = grp.dropna(subset=["day_odds"])
            grp_v = grp_v[grp_v["day_odds"] > 0]
            if len(grp_v) < 3:
                continue
            top3 = grp_v.nsmallest(3, "day_odds")
            result[rid] = (
                top3.iloc[0]["umaban"],
                top3.iloc[1]["umaban"],
                top3.iloc[2]["umaban"],
            )
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------
# レース特徴量構築 (E/F共通)
# ---------------------------------------------------------------
def _build_race_ef(te_df, horse_pred, X_te, sed_pop, def_name,
                   factor_scores=None):
    """
    E: actual_upset = fav1,2,3全員が1着にこない
    F: actual_upset = fav1,2,3全員が2着以内にこない
    """
    pred_df = pd.DataFrame({
        "race_id": te_df["race_id"].values,
        "p_target": horse_pred,
    })
    if "day_win_odds" in X_te.columns:
        pred_df["day_win_odds"] = X_te["day_win_odds"].values
    if factor_scores is not None:
        pred_df["factor_score"] = factor_scores

    results = []
    for rid, grp in pred_df.groupby("race_id"):
        row = {"race_id": rid}

        # ---- 確定人気でラベル判定 ----
        sed_r = sed_pop[sed_pop["race_id"] == rid]
        if len(sed_r) == 0:
            continue
        fav1_s = sed_r[sed_r["final_pop"] == 1]
        fav2_s = sed_r[sed_r["final_pop"] == 2]
        fav3_s = sed_r[sed_r["final_pop"] == 3]
        if len(fav1_s) == 0 or len(fav2_s) == 0 or len(fav3_s) == 0:
            continue
        f1_pos = fav1_s["finish_pos"].iloc[0]
        f2_pos = fav2_s["finish_pos"].iloc[0]
        f3_pos = fav3_s["finish_pos"].iloc[0]
        if pd.isna(f1_pos) or pd.isna(f2_pos) or pd.isna(f3_pos):
            continue

        if def_name == "E":
            row["actual_upset"] = int(
                f1_pos > 1 and f2_pos > 1 and f3_pos > 1)
        else:  # F
            row["actual_upset"] = int(
                f1_pos > 2 and f2_pos > 2 and f3_pos > 2)

        # ---- day_win_odds で fav1/fav2/fav3 を推定 ----
        grp_v = grp.dropna(subset=["day_win_odds"])
        grp_v = grp_v[grp_v["day_win_odds"] > 0]
        if len(grp_v) < 3:
            continue
        grp_sorted = grp_v.sort_values("day_win_odds")
        fav1 = grp_sorted.iloc[0]
        fav2 = grp_sorted.iloc[1]
        fav3 = grp_sorted.iloc[2]
        rest = grp_sorted.iloc[3:]

        p1, p2, p3 = (
            fav1["p_target"], fav2["p_target"], fav3["p_target"])
        o1, o2, o3 = (
            fav1["day_win_odds"], fav2["day_win_odds"],
            fav3["day_win_odds"])

        row["p_fav1"] = p1
        row["p_fav2"] = p2
        row["p_fav3"] = p3
        row["p_product"] = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
        row["fav1_odds"] = o1
        row["fav2_odds"] = o2
        row["fav3_odds"] = o3
        row["fav_implied_sum"] = (
            1.0 / max(o1, 1) + 1.0 / max(o2, 1)
            + 1.0 / max(o3, 1))
        row["fav1_implied"] = 1.0 / max(o1, 1)
        row["n_horses"] = len(grp)

        # オッズベースP(upset)
        if def_name == "E":
            # P(win) ≈ 0.8/odds
            p1_o = min(0.95, 0.8 / max(o1, 1))
            p2_o = min(0.95, 0.8 / max(o2, 1))
            p3_o = min(0.95, 0.8 / max(o3, 1))
        else:  # F
            # P(top2) ≈ 1.5/odds
            p1_o = min(0.95, 1.5 / max(o1, 1))
            p2_o = min(0.95, 1.5 / max(o2, 1))
            p3_o = min(0.95, 1.5 / max(o3, 1))
        row["p_product_odds"] = (
            (1.0 - p1_o) * (1.0 - p2_o) * (1.0 - p3_o))

        # 非人気馬の脅威度
        if len(rest) > 0:
            rp = rest["p_target"].values
            row["p_rest_max"] = rp.max()
            row["p_rest_mean"] = rp.mean()
        else:
            rp = np.array([])
            row["p_rest_max"] = 0
            row["p_rest_mean"] = 0

        # ---- ファクタースコア集約 (レースレベル) ----
        if "factor_score" in grp.columns:
            rest_fs = (rest["factor_score"].values
                       if len(rest) > 0 else np.array([0.0]))
            fav_fs = np.array([
                fav1.get("factor_score", 0),
                fav2.get("factor_score", 0),
                fav3.get("factor_score", 0),
            ])
            row["fs_rest_max"] = rest_fs.max()
            row["fs_rest_mean"] = rest_fs.mean()
            row["fs_fav_mean"] = fav_fs.mean()
            row["fs_gap"] = rest_fs.max() - fav_fs.mean()
            row["fs_rest_top2_sum"] = sum(
                sorted(rest_fs, reverse=True)[:2])
            row["fs_rest_positive_n"] = int((rest_fs > 0).sum())
            row["fs_rest_strong_n"] = int(
                (rest_fs > 0.3).sum())

        # ---- リッチ特徴量 (horse_model.py由来) ----
        fav_p = np.array([p1, p2, p3])
        all_p = grp["p_target"].values

        # 人気馬統計
        row["h_fav_mean"] = fav_p.mean()
        row["h_fav_min"] = fav_p.min()
        row["h_fav_max"] = fav_p.max()
        row["h_fav_std"] = fav_p.std()
        safe_fav = np.clip(fav_p, 0.01, 0.99)
        row["h_fav_logodds_sum"] = np.sum(
            np.log(safe_fav / (1 - safe_fav)))
        row["h_fav12_gap"] = p1 - p2
        row["h_fav_weak_count"] = int((fav_p < 0.5).sum())
        row["h_fav_very_weak"] = int((fav_p < 0.35).sum())

        # 非人気馬統計
        if len(rp) > 0:
            row["h_rest_std"] = rp.std() if len(rp) > 1 else 0
            row["h_rest_beat_fav3"] = int((rp > p3).sum())
            top2_rest = sorted(rp, reverse=True)[:2]
            row["h_rest_top2_sum"] = sum(top2_rest)
            row["h_rest_above_40"] = int((rp > 0.4).sum())
            row["h_rest_above_50"] = int((rp > 0.5).sum())
        else:
            row["h_rest_std"] = 0
            row["h_rest_beat_fav3"] = 0
            row["h_rest_top2_sum"] = 0
            row["h_rest_above_40"] = 0
            row["h_rest_above_50"] = 0

        # ギャップ・全体統計
        row["h_gap"] = fav_p.mean() - (rp.mean() if len(rp) > 0 else 0)
        row["h_all_std"] = all_p.std()
        row["h_all_range"] = all_p.max() - all_p.min()

        # エントロピー（混戦度）
        safe_all = np.clip(all_p, 0.001, 0.999)
        prob_norm = safe_all / safe_all.sum()
        row["h_pred_entropy"] = _entropy(prob_norm)

        # Gini不均衡度
        sorted_p = np.sort(all_p)
        n_all = len(sorted_p)
        if n_all > 1 and sorted_p.sum() > 0:
            idx = np.arange(1, n_all + 1)
            row["h_pred_gini"] = (
                (2.0 * np.sum(idx * sorted_p)
                 / (n_all * sorted_p.sum()))
                - (n_all + 1.0) / n_all)
        else:
            row["h_pred_gini"] = 0

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------
# 払戻事前計算 (fav3除外)
# ---------------------------------------------------------------
def _fav_in_combo(combo_str, excl_set, n_horses):
    if not combo_str or combo_str == "nan":
        return True
    s = combo_str.strip()
    if len(s) < n_horses * 2:
        return True
    horses = [s[i * 2:(i + 1) * 2] for i in range(n_horses)]
    return any(h in excl_set for h in horses)


_TANSHO_ALL = ["単勝払戻1", "単勝払戻2", "単勝払戻3"]
_FUKUSHO_ALL = ["複勝払戻1", "複勝払戻2", "複勝払戻3",
                "複勝払戻4", "複勝払戻5"]
_UMAREN_ALL = ["馬連払戻1", "馬連払戻2", "馬連払戻3"]
_UMATAN_ALL = ["馬単払戻1", "馬単払戻2", "馬単払戻3",
               "馬単払戻4", "馬単払戻5", "馬単払戻6"]
_SANRENPUKU_ALL = ["三連複払戻1", "三連複払戻2", "三連複払戻3"]
_SANRENTAN_ALL = ["三連単払戻1", "三連単払戻2", "三連単払戻3",
                  "三連単払戻4", "三連単払戻5", "三連単払戻6"]
_WIDE_ALL = ["ワイド払戻1", "ワイド払戻2", "ワイド払戻3",
             "ワイド払戻4", "ワイド払戻5", "ワイド払戻6",
             "ワイド払戻7"]

# fav3除外券種定義: (cost_fn, prefixes, n_horses_in_combo, ptype)
_EXCL3_TICKET_DEFS = {
    "単勝ex3": (
        lambda n: max(n - 3, 1) * 100,
        _TANSHO_ALL, 1, "tansho"),
    "複勝ex3": (
        lambda n: max(n - 3, 1) * 100,
        _FUKUSHO_ALL, 1, "fukusho"),
    "馬連ex3": (
        lambda n: comb(max(n - 3, 2), 2) * 100,
        _UMAREN_ALL, 2, "combo"),
    "馬単ex3": (
        lambda n: perm(max(n - 3, 2), 2) * 100,
        _UMATAN_ALL, 2, "combo"),
    "三連複ex3": (
        lambda n: comb(max(n - 3, 3), 3) * 100,
        _SANRENPUKU_ALL, 3, "combo"),
    "三連単ex3": (
        lambda n: perm(max(n - 3, 3), 3) * 100,
        _SANRENTAN_ALL, 3, "combo"),
    "ワイドex3": (
        lambda n: comb(max(n - 3, 2), 2) * 100,
        _WIDE_ALL, 2, "combo"),
}


def precompute_economics_ef(hjc_df, nh_dict, race_ids, fav_umas):
    """各レース×各券種の (cost, payout) を事前計算 (fav3除外)
    fav_umas: {race_id: (fav1, fav2, fav3)}
    """
    econ = {}
    for tname, (cost_fn, prefixes, nhc, ptype) in \
            _EXCL3_TICKET_DEFS.items():
        race_econ = {}
        for rid in race_ids:
            nh = nh_dict.get(rid, 14)
            cost = cost_fn(nh)
            f1, f2, f3 = fav_umas.get(rid, ("??", "??", "??"))
            excl_set = {f1, f2, f3}
            row = hjc_df[hjc_df["race_id"] == rid]
            payout = 0
            if not row.empty:
                if ptype == "tansho":
                    for pfx in prefixes:
                        val = row.iloc[0].get(f"{pfx}_払戻金", 0)
                        if val > 0:
                            uma = str(row.iloc[0].get(
                                f"{pfx}_馬番", "")).strip().zfill(2)
                            if uma not in excl_set:
                                payout += val
                elif ptype == "fukusho":
                    for pfx in prefixes:
                        val = row.iloc[0].get(f"{pfx}_払戻金", 0)
                        if val > 0:
                            uma = str(row.iloc[0].get(
                                f"{pfx}_馬番", "")).strip().zfill(2)
                            if uma not in excl_set:
                                payout += val
                else:  # combo
                    for pfx in prefixes:
                        val = row.iloc[0].get(f"{pfx}_払戻金", 0)
                        combo = str(row.iloc[0].get(
                            f"{pfx}_馬番組合せ", ""))
                        if val > 0 and not _fav_in_combo(
                                combo, excl_set, nhc):
                            payout += val
            race_econ[rid] = (cost, payout)
        econ[tname] = race_econ
    return econ


def fast_roi(race_econ, selected_rids):
    total_cost = 0
    total_ret = 0
    for rid in selected_rids:
        if rid in race_econ:
            c, p = race_econ[rid]
            total_cost += c
            total_ret += p
    if total_cost == 0:
        return 0
    return total_ret / total_cost * 100


def fast_roi_detail(race_econ, selected_rids):
    """ROI + 的中数 + 投資/払戻 詳細"""
    total_cost = 0
    total_ret = 0
    n_hit = 0
    for rid in selected_rids:
        if rid in race_econ:
            c, p = race_econ[rid]
            total_cost += c
            total_ret += p
            if p > 0:
                n_hit += 1
    n_races = len(selected_rids)
    roi = total_ret / total_cost * 100 if total_cost > 0 else 0
    return {
        "roi": roi, "n_races": n_races, "n_hit": n_hit,
        "hit_rate": n_hit / n_races if n_races > 0 else 0,
        "total_cost": total_cost, "total_ret": total_ret,
        "profit": total_ret - total_cost,
    }


# ---------------------------------------------------------------
# スコアリング
# ---------------------------------------------------------------
def compute_scores(rf):
    p = rf["p_product"].values
    po = rf["p_product_odds"].values
    fis = rf["fav_implied_sum"].values
    fi1 = rf["fav1_implied"].values
    nh = rf["n_horses"].values.astype(float)
    prm = rf["p_rest_max"].values
    scores = {
        "naive": p,
        "odds_naive": po,
        "blend50": 0.5 * p + 0.5 * po,
        "blend70": 0.7 * p + 0.3 * po,
        "payout_w": p * fis,
        "payout_f1": p * fi1,
        "field_adj": p / np.sqrt(np.maximum(nh, 1)),
        "rest_threat": p * prm,
        "combo": p * fis * prm,
        "ev_excl": (
            p * fis
            / np.maximum((nh - 3) * (nh - 4) / 2, 1) * 100),
    }
    # Stage 2 スコア
    if "stage2_score" in rf.columns:
        s2 = rf["stage2_score"].values
        scores["s2_pure"] = s2
        scores["s2_blend50"] = 0.5 * s2 + 0.5 * p
        scores["s2_blend70"] = 0.7 * s2 + 0.3 * p
        scores["s2_x_odds"] = s2 * fis
        scores["s2_x_rest"] = s2 * prm
    # ファクター系スコア
    if "fs_rest_max" in rf.columns:
        fs_rm = rf["fs_rest_max"].values
        fs_gap = rf["fs_gap"].values
        scores["factor_pure"] = fs_rm
        scores["p_x_factor"] = p * np.maximum(fs_rm, 0.01)
        scores["factor_gap"] = fs_gap
        scores["combo_factor"] = p * fis * np.maximum(fs_rm, 0.01)
    if "stage2_score" in rf.columns and "fs_rest_max" in rf.columns:
        s2 = rf["stage2_score"].values
        scores["s2_x_factor"] = s2 * np.maximum(fs_rm, 0.01)
        scores["s2_blend_factor"] = (
            0.5 * s2 + 0.3 * p + 0.2 * fs_rm)
    # リッチ特徴量スコア
    if "h_pred_entropy" in rf.columns:
        ent = rf["h_pred_entropy"].values
        scores["entropy_w"] = p * ent
    if "h_fav_logodds_sum" in rf.columns:
        logodds = rf["h_fav_logodds_sum"].values
        scores["logodds_inv"] = p * np.maximum(-logodds, 0)
    return scores


# ---------------------------------------------------------------
# グリッドサーチ
# ---------------------------------------------------------------
def grid_search(race_data, year_scores, econ, test_years,
                ticket_names, min_races=3):
    score_names = list(year_scores[test_years[0]].keys())
    pct_list = list(range(1, 21)) + [25, 30]
    ff_list = [99, 16, 14, 12, 10]
    fav1_odds_list = [99, 3.0, 2.5, 2.0, 1.5]

    n_total = (
        len(score_names) * len(pct_list) * len(ff_list)
        * len(fav1_odds_list) * len(ticket_names))
    print(f"  探索空間: ~{n_total:,}通り")
    sys.stdout.flush()

    winners = []
    all_candidates = []

    for sn in score_names:
        for ff in ff_list:
            for fo in fav1_odds_list:
                for pct in pct_list:
                    year_sels = {}
                    skip = False
                    for yr in test_years:
                        rf = race_data[yr]
                        mask = np.ones(len(rf), dtype=bool)
                        if ff < 99:
                            mask &= (rf["n_horses"].values <= ff)
                        if fo < 99:
                            mask &= (rf["fav1_odds"].values <= fo)

                        all_s = year_scores[yr][sn]
                        filt_s = all_s[mask]
                        if len(filt_s) < min_races:
                            skip = True
                            break
                        cutoff = np.percentile(filt_s, 100 - pct)
                        sel_mask = mask & (all_s >= cutoff)
                        rids = (
                            rf["race_id"].values[sel_mask].tolist())
                        if len(rids) < min_races:
                            skip = True
                            break
                        year_sels[yr] = rids

                    if skip:
                        continue

                    for tname in ticket_names:
                        if tname not in econ[test_years[0]]:
                            continue
                        rois, ns, hits = [], [], []
                        for yr in test_years:
                            d = fast_roi_detail(
                                econ[yr][tname], year_sels[yr])
                            rois.append(d["roi"])
                            ns.append(d["n_races"])
                            hits.append(d["n_hit"])

                        total_hit = sum(hits)
                        total_n = sum(ns)
                        entry = {
                            "score": sn, "type": f"top{pct}%",
                            "ff": ff, "fav1_o": fo,
                            "ticket": tname,
                            "avg_roi": np.mean(rois),
                            "min_roi": min(rois),
                            "avg_n": np.mean(ns),
                            "avg_hit_rate": (
                                total_hit / total_n
                                if total_n > 0 else 0),
                        }
                        for i, yr in enumerate(test_years):
                            entry[f"roi_{yr}"] = rois[i]
                            entry[f"n_{yr}"] = ns[i]
                            entry[f"hit_{yr}"] = hits[i]

                        mn = min(rois)
                        mr = min(ns)
                        if mn > 100 and mr >= min_races:
                            winners.append(entry)
                        nw = sum(1 for r in rois if r > 100)
                        if (nw >= len(test_years) - 1
                                and mr >= min_races):
                            all_candidates.append(entry)

    return winners, all_candidates


# ---------------------------------------------------------------
# メイン
# ---------------------------------------------------------------
def main():
    start = time.time()
    print("=" * 70)
    print("  E/F 荒れ予測 & ROI最適化")
    if L1_ONLY:
        print("  ★ L1-onlyモード (LGB/CB skip)")
    print("  E: fav1,2,3が1着にこない → 単勝ex3")
    print("  F: fav1,2,3が1,2着にこない → 馬単ex3")
    print("=" * 70)

    # ---- Phase 1: 特徴量構築 + ファクター統合 ----
    print("\n[1] 馬特徴量構築 (return_raw=True)...")
    sys.stdout.flush()
    df = build_horse_features(
        feature_groups=BEST_FEATURE_GROUPS, return_raw=True)

    # 追加SED列 (不利/タイム等 — extra=True ファクター用)
    print("  追加SED列をロード中...")
    extra = _load_extra_prev_cols()
    existing = set(df.columns)
    extra_cols = [c for c in extra.columns
                  if c not in existing or c == "horse_race_id"]
    df = df.merge(extra[extra_cols], on="horse_race_id", how="left")

    # 68ファクターを計算
    print("  ファクター計算中...")
    factor_cols = compute_factors(df)
    print(f"  {len(factor_cols)}ファクター計算完了")

    # ファクター二値列 + ターゲットを保存 (フォールド内L1推定用)
    _fp = _to_numeric(df["finish_pos"])
    _fpop = _to_numeric(df["final_pop"])
    factor_binary_store = df[["race_id", "date"] + factor_cols].copy()
    factor_binary_store["upset_horse"] = (
        _fp.notna() & _fpop.notna() & (_fpop > 3) & (_fp <= 3)
    ).astype(int)

    # factor_score はフォールド毎に動的計算 (仮値0)
    df["factor_score"] = 0.0

    # 生列 + 個別ファクター列をドロップ (ML特徴量から除外)
    drop_set = set(_RAW_DROP + factor_cols)
    df = df.drop(
        columns=[c for c in drop_set if c in df.columns],
        errors="ignore")

    df, cat_cols = _prepare(df)
    feat_cols = sorted([c for c in df.columns if c not in DROP])
    cat_indices = [feat_cols.index(c) for c in cat_cols
                   if c in feat_cols]
    print(f"  {len(df)}行, {len(feat_cols)}特徴量"
          f" (factor_score含む), cat={len(cat_indices)}")
    sys.stdout.flush()

    year = df["date"].str[:4]
    rid_to_date = df.groupby("race_id")["date"].first().to_dict()

    # ---- Phase 2: SED確定人気データ ----
    print("\n[2] SED確定人気データ取得...")
    sys.stdout.flush()
    all_rids = df["race_id"].unique().tolist()
    sed_pop = _get_final_pop(all_rids)
    print(f"  {len(sed_pop)}件")
    sys.stdout.flush()

    # ---- Rolling configs (7年スライディングウィンドウ) ----
    configs = [
        ([str(y) for y in range(2013, 2020)], "2020", 7),
        ([str(y) for y in range(2014, 2021)], "2021", 7),
        ([str(y) for y in range(2015, 2022)], "2022", 7),
        ([str(y) for y in range(2016, 2023)], "2023", 7),
        ([str(y) for y in range(2017, 2024)], "2024", 7),
        ([str(y) for y in range(2018, 2025)], "2025", 7),
    ]
    TEST_YEARS = [c[1] for c in configs]

    ticket_names = list(_EXCL3_TICKET_DEFS.keys())

    # ---- E/F それぞれ評価 ----
    for def_name, target_col, primary_ticket in [
        ("E", "win", "単勝ex3"),
        ("F", "top2_finish", "馬単ex3"),
    ]:
        print(f"\n{'='*70}")
        print(f"  === 定義{def_name} ===")
        if def_name == "E":
            print("  荒れ: fav1,2,3が全員1着にこない")
            print("  馬モデル: P(win) = P(着順==1)")
        else:
            print("  荒れ: fav1,2,3が全員2着以内にこない")
            print("  馬モデル: P(top2) = P(着順<=2)")
        print(f"  主要券種: {primary_ticket}")
        print(f"{'='*70}")

        # ---- 馬モデル学習 ----
        if L1_ONLY:
            print(f"\n  [3-{def_name}] L1-only モード"
                  " (LGB/CB skip, odds-implied + factor_score)...")
        else:
            print(f"\n  [3-{def_name}] 馬モデル学習"
                  " (LGB+CB+Calibration)...")
        sys.stdout.flush()

        race_data = {}
        for tr_yrs, te_yr, n_seeds in configs:
            label = f"{tr_yrs[0]}-{tr_yrs[-1]}→{te_yr}"
            print(f"\n    {label}")
            sys.stdout.flush()

            # --- L1重みをフォールド内推定 (tr_yrsと同期間) ---
            l1_yrs = tr_yrs
            fold_weights = _fit_l1_weights(
                factor_binary_store, factor_cols, l1_yrs)
            n_nz = len(fold_weights)
            print(f"    L1重み: {n_nz}個非ゼロ"
                  f" (学習: {l1_yrs[0]}-{l1_yrs[-1]})")

            # factor_score を全行更新 (このフォールドの重みで)
            weight_vec = np.array(
                [fold_weights.get(fc, 0) for fc in factor_cols])
            df["factor_score"] = (
                factor_binary_store[factor_cols]
                .fillna(0).values @ weight_vec)

            tr = df[year.isin(tr_yrs)].reset_index(drop=True)
            te = df[year == te_yr].reset_index(drop=True)
            X_tr, y_tr = tr[feat_cols], tr[target_col]
            X_te, y_te = te[feat_cols], te[target_col]

            if not L1_ONLY:
                # LGB ensemble
                lgb_pred = _train_lgb(
                    X_tr, y_tr, X_te, y_te, n=n_seeds)
                auc_lgb = roc_auc_score(y_te, lgb_pred)
                print(f"    LGB AUC: {auc_lgb:.4f}")
                sys.stdout.flush()

                # CatBoost ensemble
                cb_pred = _train_cb(
                    X_tr, y_tr, X_te, y_te, cat_indices, n=N_ENS_CB)
                auc_cb = roc_auc_score(y_te, cb_pred)
                print(f"    CB  AUC: {auc_cb:.4f}")
                sys.stdout.flush()

                # Isotonic calibration
                cal_year = tr_yrs[-1]
                cal_tr_yrs = tr_yrs[:-1]
                if len(cal_tr_yrs) >= 1:
                    cal_tr = df[year.isin(cal_tr_yrs)].reset_index(
                        drop=True)
                    cal_te = df[year == cal_year].reset_index(
                        drop=True)
                    X_cal_tr = cal_tr[feat_cols]
                    y_cal_tr = cal_tr[target_col]
                    X_cal_te = cal_te[feat_cols]
                    y_cal_te = cal_te[target_col]
                    lgb_cal = _train_lgb(
                        X_cal_tr, y_cal_tr, X_cal_te, y_cal_te, n=5)
                    cb_cal = _train_cb(
                        X_cal_tr, y_cal_tr, X_cal_te, y_cal_te,
                        cat_indices, n=N_ENS_CB)
                    pred = _blend_and_calibrate(
                        lgb_cal, cb_cal, y_cal_te,
                        lgb_pred, cb_pred)
                else:
                    pred = (BLEND_WEIGHT_LGB * lgb_pred
                            + (1 - BLEND_WEIGHT_LGB) * cb_pred)
                auc_cal = roc_auc_score(y_te, pred)
                print(f"    Calibrated AUC: {auc_cal:.4f}")
            else:
                # L1-only: オッズ由来の暗黙確率を使用
                odds_col = "day_win_odds"
                if odds_col in X_te.columns:
                    odds = X_te[odds_col].values.astype(float)
                    if def_name == "E":
                        pred = np.clip(
                            0.8 / np.maximum(odds, 1), 0.01, 0.95)
                    else:  # F
                        pred = np.clip(
                            1.5 / np.maximum(odds, 1), 0.01, 0.95)
                else:
                    pred = np.full(len(X_te), 0.5)
                print(f"    L1-only: odds-implied pred"
                      f" (skip LGB/CB)")
                sys.stdout.flush()

            # ファクタースコアをレースレベル集約に渡す
            te_mask = (factor_binary_store["date"].str[:4] == te_yr)
            te_fs = (factor_binary_store.loc[te_mask, factor_cols]
                     .fillna(0).values @ weight_vec)
            fs_vals = te_fs if len(te_fs) == len(te) else None
            rf = _build_race_ef(
                te, pred, X_te, sed_pop, def_name,
                factor_scores=fs_vals)
            race_data[te_yr] = rf

            n_upset = int(rf["actual_upset"].sum())
            pct = n_upset / len(rf) if len(rf) > 0 else 0
            print(f"    レース: {len(rf)},"
                  f" 荒れ: {n_upset} ({pct:.1%})")

            if rf["actual_upset"].nunique() > 1:
                naive_auc = roc_auc_score(
                    rf["actual_upset"], rf["p_product"])
                odds_auc = roc_auc_score(
                    rf["actual_upset"], rf["p_product_odds"])
                print(f"    レースAUC: naive={naive_auc:.4f},"
                      f" odds={odds_auc:.4f}")
            sys.stdout.flush()

        # ---- Stage 2: 荒れ予測モデル ----
        print(f"\n  [S2-{def_name}] Stage2 荒れ予測モデル...")
        sys.stdout.flush()
        s2_feat_cols = sorted([
            c for c in race_data[TEST_YEARS[0]].columns
            if c not in ("race_id", "actual_upset")])
        for te_yr_idx, te_yr in enumerate(TEST_YEARS):
            prior_years = TEST_YEARS[:te_yr_idx]
            rf = race_data[te_yr]
            if len(prior_years) == 0:
                rf["stage2_score"] = rf["p_product"]
                print(f"    {te_yr}: Stage2 skip (prior=0), "
                      f"fallback to p_product")
                continue
            s2_train = pd.concat(
                [race_data[y] for y in prior_years],
                ignore_index=True)
            X_s2_tr = s2_train[s2_feat_cols]
            y_s2_tr = s2_train["actual_upset"]
            X_s2_te = rf[s2_feat_cols]
            neg2 = int((y_s2_tr == 0).sum())
            pos2 = int((y_s2_tr == 1).sum())
            s2_preds = []
            for i in range(10):
                s2_p = {
                    "objective": "binary", "metric": "auc",
                    "learning_rate": 0.05, "num_leaves": 15,
                    "max_depth": 4, "min_child_samples": 30,
                    "subsample": 0.8, "colsample_bytree": 0.7,
                    "reg_alpha": 0.1, "reg_lambda": 0.1,
                    "scale_pos_weight": neg2 / max(pos2, 1),
                    "verbose": -1, "n_jobs": -1,
                    "seed": 42 + i * 7,
                }
                ds = lgb.Dataset(X_s2_tr, label=y_s2_tr)
                m = lgb.train(
                    params=s2_p, train_set=ds, num_boost_round=200)
                s2_preds.append(m.predict(X_s2_te))
            rf["stage2_score"] = np.median(s2_preds, axis=0)
            race_data[te_yr] = rf
            if rf["actual_upset"].nunique() > 1:
                s2_auc = roc_auc_score(
                    rf["actual_upset"], rf["stage2_score"])
                print(f"    {te_yr}: Stage2 AUC={s2_auc:.4f}"
                      f" (train={len(s2_train)}R)")
            else:
                print(f"    {te_yr}: Stage2 done"
                      f" (train={len(s2_train)}R)")
            sys.stdout.flush()

        # ---- 払戻データ ----
        print(f"\n  [4-{def_name}] 払戻データ + fav3馬番取得...")
        sys.stdout.flush()
        econ = {}
        for te_yr in TEST_YEARS:
            rids = race_data[te_yr]["race_id"].tolist()
            hjc_df = _get_hjc_full(rids)
            nh_dict = _get_n_horses(rids)
            fav_umas = _get_fav_umabans_top3(rids)
            econ[te_yr] = precompute_economics_ef(
                hjc_df, nh_dict, rids, fav_umas)
            print(f"    {te_yr}: {len(fav_umas)} fav馬番取得")
        sys.stdout.flush()

        # ---- 理論上限 ----
        print(f"\n  [5-{def_name}] 理論上限 (完璧予測)")
        for te_yr in TEST_YEARS:
            rf = race_data[te_yr]
            upset_rids = (
                rf[rf["actual_upset"] == 1]["race_id"].tolist())
            total = len(rf)
            print(f"\n    {te_yr}: {len(upset_rids)} 荒れ"
                  f" / {total} 全"
                  f" ({len(upset_rids)/total:.1%})")
            for tname in ticket_names:
                if tname in econ[te_yr]:
                    roi = fast_roi(
                        econ[te_yr][tname], upset_rids)
                    marker = " ★" if tname == primary_ticket else ""
                    print(f"      {tname:>8s}:"
                          f" {roi:6.1f}%{marker}")

        # ---- スコア計算 ----
        year_scores = {}
        for te_yr in TEST_YEARS:
            year_scores[te_yr] = compute_scores(
                race_data[te_yr])

        # ---- グリッドサーチ ----
        print(f"\n  [6-{def_name}] グリッドサーチ")
        sys.stdout.flush()
        winners, candidates = grid_search(
            race_data, year_scores, econ, TEST_YEARS,
            ticket_names)

        # ---- 結果レポート ----
        print(f"\n  [7-{def_name}] 結果")
        n_yrs = len(TEST_YEARS)

        if winners:
            w_df = pd.DataFrame(winners).sort_values(
                "min_roi", ascending=False)
            print(f"\n    ★★★ {len(w_df)}件"
                  f" {n_yrs}年間ROI>100%達成! ★★★\n")

            for tname in ticket_names:
                sub = w_df[w_df["ticket"] == tname]
                if len(sub) == 0:
                    continue
                print(f"\n    [{tname}] {len(sub)}件")
                hdr = (f"    {'#':>2} {'score':>10}"
                       f" {'sel':>7} {'nh':>4} {'fo':>4} |")
                for yr in TEST_YEARS:
                    hdr += f" {yr:>16}"
                hdr += f" | {'avg':>5} {'min':>5} {'hit%':>5}"
                print(hdr)
                print("    " + "-" * (59 + 17 * n_yrs))
                for i, (_, r) in enumerate(
                        sub.head(15).iterrows()):
                    ff = (f"≤{r['ff']:.0f}"
                          if r['ff'] < 99 else " -")
                    fo = (f"≤{r['fav1_o']:.1f}"
                          if r['fav1_o'] < 99 else " -")
                    line = (
                        f"    {i+1:2d} {r['score']:>10s}"
                        f" {r['type']:>7s}"
                        f" {ff:>4s} {fo:>4s} |")
                    for yr in TEST_YEARS:
                        h = r.get(f'hit_{yr}', 0)
                        line += (
                            f" {r[f'roi_{yr}']:5.1f}%"
                            f"({r[f'n_{yr}']:3.0f}R"
                            f"/{h:.0f}h)")
                    hr = r.get('avg_hit_rate', 0)
                    line += (f" | {r['avg_roi']:5.1f}%"
                             f" {r['min_roi']:5.1f}%"
                             f" {hr:4.0%}")
                    print(line)
        else:
            print(f"\n    {n_yrs}年間全てROI>100%の戦略は"
                  "見つかりませんでした")

        # ---- Near-miss (3年以上 ROI>100%) ----
        if candidates:
            c_df = pd.DataFrame(candidates)
            c_df["n_win"] = sum(
                (c_df[f"roi_{yr}"] > 100).astype(int)
                for yr in TEST_YEARS)
            c_df = c_df.sort_values(
                ["n_win", "min_roi"], ascending=[False, False])

            print(f"\n    --- 惜しい候補"
                  f" ({n_yrs - 1}年以上ROI>100%) ---")
            for tname in ticket_names:
                sub = c_df[c_df["ticket"] == tname].head(5)
                if len(sub) == 0:
                    continue
                print(f"\n    [{tname}]")
                for _, r in sub.iterrows():
                    ff = (f"nh≤{r['ff']:.0f}"
                          if r['ff'] < 99 else "全")
                    fo = (f"fo≤{r['fav1_o']:.1f}"
                          if r['fav1_o'] < 99 else "")
                    marks = "".join(
                        "O" if r[f"roi_{yr}"] > 100 else "X"
                        for yr in TEST_YEARS)
                    parts = " ".join(
                        f"{yr[2:]}={r[f'roi_{yr}']:5.1f}%"
                        f"({r[f'n_{yr}']:3.0f}R"
                        f"/{r.get(f'hit_{yr}', 0):.0f}h)"
                        for yr in TEST_YEARS)
                    print(
                        f"    {r['score']:>10s}"
                        f" {r['type']:>7s}"
                        f" {ff:>5s} {fo:>7s} |"
                        f" {parts}"
                        f" [{marks}]")

        # ---- ベスト戦略サマリ ----
        if winners:
            best = max(winners, key=lambda x: x["min_roi"])
            print(f"\n    ★ 定義{def_name} ベスト戦略:")
            ff_s = (f" nh≤{int(best['ff'])}"
                    if best['ff'] < 99 else "")
            fo_s = (f" fo≤{best['fav1_o']}"
                    if best['fav1_o'] < 99 else "")
            print(f"      {best['score']}"
                  f" {best['type']}{ff_s}{fo_s}"
                  f" {best['ticket']}")
            print(f"      平均ROI: {best['avg_roi']:.1f}%"
                  f"  最低ROI: {best['min_roi']:.1f}%"
                  f"  的中率: {best['avg_hit_rate']:.0%}")

            # 年別詳細
            tname = best["ticket"]
            print(f"\n      {'年':>6s} {'ROI':>7s}"
                  f" {'レース':>5s} {'的中':>4s}"
                  f" {'的中率':>6s}"
                  f" {'投資':>10s} {'払戻':>10s}"
                  f" {'利益':>10s}")
            print(f"      " + "-" * 65)

            sum_cost, sum_ret, sum_n, sum_hit = 0, 0, 0, 0
            for yr in TEST_YEARS:
                # 再計算して詳細を取得
                rf = race_data[yr]
                sc = year_scores[yr]
                sn = best["score"]
                pct = int(best["type"].replace("top", "")
                          .replace("%", ""))
                mask = np.ones(len(rf), dtype=bool)
                if best["ff"] < 99:
                    mask &= (rf["n_horses"].values
                             <= best["ff"])
                if best["fav1_o"] < 99:
                    mask &= (rf["fav1_odds"].values
                             <= best["fav1_o"])
                all_s = sc[sn]
                filt_s = all_s[mask]
                cutoff = np.percentile(filt_s, 100 - pct)
                sel_mask = mask & (all_s >= cutoff)
                rids = rf["race_id"].values[sel_mask].tolist()
                d = fast_roi_detail(
                    econ[yr][tname], rids)
                sum_cost += d["total_cost"]
                sum_ret += d["total_ret"]
                sum_n += d["n_races"]
                sum_hit += d["n_hit"]
                print(
                    f"      {yr:>6s}"
                    f" {d['roi']:6.1f}%"
                    f" {d['n_races']:5d}"
                    f" {d['n_hit']:4d}"
                    f" {d['hit_rate']:5.0%}"
                    f" {d['total_cost']:>10,}円"
                    f" {d['total_ret']:>10,}円"
                    f" {d['profit']:>+10,}円")
            # 合計
            total_roi = (sum_ret / sum_cost * 100
                         if sum_cost > 0 else 0)
            total_hr = (sum_hit / sum_n
                        if sum_n > 0 else 0)
            print(f"      " + "-" * 65)
            print(
                f"      {'合計':>6s}"
                f" {total_roi:6.1f}%"
                f" {sum_n:5d}"
                f" {sum_hit:4d}"
                f" {total_hr:5.0%}"
                f" {sum_cost:>10,}円"
                f" {sum_ret:>10,}円"
                f" {sum_ret - sum_cost:>+10,}円")

        # ---- 詳細データ保存 (月別分析用) ----
        os.makedirs("output", exist_ok=True)
        detail = {
            "race_data": race_data,
            "econ": econ,
            "year_scores": year_scores,
            "rid_to_date": rid_to_date,
            "test_years": TEST_YEARS,
            "winners": winners,
            "candidates": candidates,
        }
        pkl_path = os.path.join("output", f"{def_name}_detail.pkl")
        pickle.dump(detail, open(pkl_path, "wb"))
        print(f"\n  詳細データを {pkl_path} に保存")

    elapsed = time.time() - start
    print(f"\n\n完了! ({elapsed:.1f}秒)")


if __name__ == "__main__":
    main()
