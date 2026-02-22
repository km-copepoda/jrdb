"""
E定義 ローリング予測 → 各券種の利益計算 (2024-2026)

E: fav1,2,3が全員1着にこない → モデルで予測
券種: 馬単, 馬連, 三連単, 三連複 (全通り / ex2 / ex3)
ローリング:
  2018-2023→2024, 2018-2024→2025, 2018-2025→2026
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
import psycopg2
import lightgbm as lgb
from math import comb, perm
from sklearn.metrics import roc_auc_score
from horse_model import build_horse_features, _to_numeric, _prepare
from config import DB_CONFIG, T_BAC, T_HJC, T_SED, T_KYI, T_TYB

warnings.filterwarnings("ignore")

DROP = ["race_id", "top3_finish", "win", "top2_finish", "date"]

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
        for c in [c for c in cols if "組合せ" in c]:
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


def _build_race_e(te_df, horse_pred, X_te, sed_pop):
    """E定義用レース特徴量"""
    pred_df = pd.DataFrame({
        "race_id": te_df["race_id"].values,
        "p_win": horse_pred,
    })
    if "day_win_odds" in X_te.columns:
        pred_df["day_win_odds"] = X_te["day_win_odds"].values

    results = []
    for rid, grp in pred_df.groupby("race_id"):
        row = {"race_id": rid}

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

        row["actual_upset"] = int(
            f1_pos > 1 and f2_pos > 1 and f3_pos > 1)

        grp_v = grp.dropna(subset=["day_win_odds"])
        grp_v = grp_v[grp_v["day_win_odds"] > 0]
        if len(grp_v) < 3:
            continue
        grp_sorted = grp_v.sort_values("day_win_odds")
        fav1 = grp_sorted.iloc[0]
        fav2 = grp_sorted.iloc[1]
        fav3 = grp_sorted.iloc[2]
        rest = grp_sorted.iloc[3:]

        p1, p2, p3 = fav1["p_win"], fav2["p_win"], fav3["p_win"]
        o1, o2, o3 = (fav1["day_win_odds"], fav2["day_win_odds"],
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

        p1_o = min(0.95, 0.8 / max(o1, 1))
        p2_o = min(0.95, 0.8 / max(o2, 1))
        p3_o = min(0.95, 0.8 / max(o3, 1))
        row["p_product_odds"] = (
            (1.0 - p1_o) * (1.0 - p2_o) * (1.0 - p3_o))

        if len(rest) > 0:
            rp = rest["p_win"].values
            row["p_rest_max"] = rp.max()
            row["p_rest_mean"] = rp.mean()
        else:
            row["p_rest_max"] = 0
            row["p_rest_mean"] = 0

        results.append(row)

    return pd.DataFrame(results)


def compute_scores(rf):
    p = rf["p_product"].values
    po = rf["p_product_odds"].values
    fis = rf["fav_implied_sum"].values
    fi1 = rf["fav1_implied"].values
    nh = rf["n_horses"].values.astype(float)
    prm = rf["p_rest_max"].values
    return {
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


def combo_has_any(combo_str, excl_set, nhc):
    if not combo_str or combo_str == "nan":
        return True
    s = combo_str.strip()
    if len(s) < nhc * 2:
        return True
    horses = [s[i * 2:(i + 1) * 2] for i in range(nhc)]
    return any(h in excl_set for h in horses)


def compute_ticket_pnl(selected_rids, hjc_df, nh_dict, fav_umas):
    """選択レースに対する各券種の損益を計算"""
    tickets = {
        "馬連(全通り)": [0, 0],
        "馬連ex2": [0, 0],
        "馬連ex3": [0, 0],
        "馬単(全通り)": [0, 0],
        "馬単ex2": [0, 0],
        "馬単ex3": [0, 0],
        "三連複(全通り)": [0, 0],
        "三連複ex2": [0, 0],
        "三連複ex3": [0, 0],
        "三連単(全通り)": [0, 0],
        "三連単ex2": [0, 0],
        "三連単ex3": [0, 0],
    }

    for rid in selected_rids:
        nh = nh_dict.get(rid, 14)
        favs = fav_umas.get(rid, ("??", "??", "??"))
        excl2 = set(favs[:2])
        excl3 = set(favs[:3])
        row = hjc_df[hjc_df["race_id"] == rid]

        # コスト加算（払戻なしでも）
        n2 = max(nh - 2, 2)
        n3 = max(nh - 3, 2)
        tickets["馬連(全通り)"][0] += comb(nh, 2) * 100
        tickets["馬連ex2"][0] += comb(n2, 2) * 100
        tickets["馬連ex3"][0] += comb(n3, 2) * 100
        tickets["馬単(全通り)"][0] += perm(nh, 2) * 100
        tickets["馬単ex2"][0] += perm(n2, 2) * 100
        tickets["馬単ex3"][0] += perm(n3, 2) * 100
        tickets["三連複(全通り)"][0] += comb(nh, 3) * 100
        tickets["三連複ex2"][0] += comb(n2, 3) * 100
        tickets["三連複ex3"][0] += comb(n3, 3) * 100
        tickets["三連単(全通り)"][0] += perm(nh, 3) * 100
        tickets["三連単ex2"][0] += perm(n2, 3) * 100
        tickets["三連単ex3"][0] += perm(n3, 3) * 100

        if row.empty:
            continue

        r = row.iloc[0]

        # 馬単系 (2頭順序あり)
        for pfx in ["馬単払戻1", "馬単払戻2", "馬単払戻3",
                     "馬単払戻4", "馬単払戻5", "馬単払戻6"]:
            val = r.get(f"{pfx}_払戻金", 0)
            if val <= 0:
                continue
            combo = str(r.get(f"{pfx}_馬番組合せ", ""))
            tickets["馬単(全通り)"][1] += val
            if not combo_has_any(combo, excl2, 2):
                tickets["馬単ex2"][1] += val
            if not combo_has_any(combo, excl3, 2):
                tickets["馬単ex3"][1] += val

        # 馬連系 (2頭順不同)
        for pfx in ["馬連払戻1", "馬連払戻2", "馬連払戻3"]:
            val = r.get(f"{pfx}_払戻金", 0)
            if val <= 0:
                continue
            combo = str(r.get(f"{pfx}_馬番組合せ", ""))
            tickets["馬連(全通り)"][1] += val
            if not combo_has_any(combo, excl2, 2):
                tickets["馬連ex2"][1] += val
            if not combo_has_any(combo, excl3, 2):
                tickets["馬連ex3"][1] += val

        # 三連複系 (3頭順不同)
        for pfx in ["三連複払戻1", "三連複払戻2", "三連複払戻3"]:
            val = r.get(f"{pfx}_払戻金", 0)
            if val <= 0:
                continue
            combo = str(r.get(f"{pfx}_馬番組合せ", ""))
            tickets["三連複(全通り)"][1] += val
            if not combo_has_any(combo, excl3, 3):
                tickets["三連複ex3"][1] += val
            if not combo_has_any(combo, excl2, 3):
                tickets["三連複ex2"][1] += val

        # 三連単系 (3頭順序あり)
        for pfx in ["三連単払戻1", "三連単払戻2", "三連単払戻3",
                     "三連単払戻4", "三連単払戻5", "三連単払戻6"]:
            val = r.get(f"{pfx}_払戻金", 0)
            if val <= 0:
                continue
            combo = str(r.get(f"{pfx}_馬番組合せ", ""))
            tickets["三連単(全通り)"][1] += val
            if not combo_has_any(combo, excl2, 3):
                tickets["三連単ex2"][1] += val
            if not combo_has_any(combo, excl3, 3):
                tickets["三連単ex3"][1] += val

    return tickets


def main():
    start = time.time()
    print("=" * 70)
    print("  E定義 ローリング予測 → 馬単/馬連 利益計算")
    print("  E: fav1,2,3が全員1着にこない")
    print("  馬モデル: P(win)")
    print("  2024-2026 ローリング")
    print("=" * 70)

    # ---- 特徴量構築 ----
    print("\n[1] 馬特徴量構築...")
    sys.stdout.flush()
    df = build_horse_features()
    df, cat_cols = _prepare(df)
    feat_cols = sorted([c for c in df.columns if c not in DROP])
    print(f"  {len(df)}行, {len(feat_cols)}特徴量")

    year = df["date"].str[:4]

    # ---- SED ----
    print("\n[2] SED確定人気データ取得...")
    sys.stdout.flush()
    all_rids = df["race_id"].unique().tolist()
    sed_pop = _get_final_pop(all_rids)
    print(f"  {len(sed_pop)}件")

    # ---- ローリング学習 ----
    configs = [
        (["2018", "2019", "2020", "2021", "2022", "2023"],
         "2024", 7),
        (["2018", "2019", "2020", "2021", "2022", "2023",
          "2024"], "2025", 7),
        (["2018", "2019", "2020", "2021", "2022", "2023",
          "2024", "2025"], "2026", 7),
    ]
    TEST_YEARS = [c[1] for c in configs]

    print("\n[3] E馬モデル学習 (P(win))...")
    sys.stdout.flush()

    race_data = {}
    for tr_yrs, te_yr, n_seeds in configs:
        label = f"{tr_yrs[0]}-{tr_yrs[-1]}→{te_yr}"
        print(f"\n  {label} (seed={n_seeds})")
        sys.stdout.flush()

        tr = df[year.isin(tr_yrs)].reset_index(drop=True)
        te = df[year == te_yr].reset_index(drop=True)
        X_tr, y_tr = tr[feat_cols], tr["win"]
        X_te, y_te = te[feat_cols], te["win"]

        pred = _train_lgb(X_tr, y_tr, X_te, y_te, n=n_seeds)
        auc = roc_auc_score(y_te, pred)
        print(f"  馬AUC (win): {auc:.4f}  正例率: {y_te.mean():.1%}")

        rf = _build_race_e(te, pred, X_te, sed_pop)
        race_data[te_yr] = rf

        n_upset = int(rf["actual_upset"].sum())
        print(f"  レース: {len(rf)},"
              f" 荒れ: {n_upset} ({n_upset/len(rf):.1%})")

        if rf["actual_upset"].nunique() > 1:
            naive_auc = roc_auc_score(
                rf["actual_upset"], rf["p_product"])
            print(f"  レースAUC: {naive_auc:.4f}")
        sys.stdout.flush()

    # ---- 払戻データ ----
    print("\n[4] 払戻データ + fav3馬番取得...")
    sys.stdout.flush()
    econ_data = {}
    for te_yr in TEST_YEARS:
        rids = race_data[te_yr]["race_id"].tolist()
        hjc_df = _get_hjc_full(rids)
        nh_dict = _get_n_horses(rids)
        fav_umas = _get_fav_umabans_top3(rids)
        econ_data[te_yr] = {
            "hjc": hjc_df, "nh": nh_dict, "fav": fav_umas}
        print(f"  {te_yr}: {len(fav_umas)} fav馬番取得")

    # ---- スコア計算 ----
    year_scores = {}
    for te_yr in TEST_YEARS:
        year_scores[te_yr] = compute_scores(race_data[te_yr])

    # ---- 複数戦略で評価 ----
    strategies = [
        ("blend70 top3% nh≤14 fo≤3.0", "blend70", 3, 14, 3.0),
        ("rest_threat top13% nh≤16 fo≤1.5", "rest_threat", 13, 16, 1.5),
        ("blend50 top5% nh≤16 fo≤3.0", "blend50", 5, 16, 3.0),
        ("blend70 top5% nh≤14 fo≤3.0", "blend70", 5, 14, 3.0),
        ("naive top3% nh≤14 fo≤3.0", "naive", 3, 14, 3.0),
        ("odds_naive top5% nh≤14 fo≤3.0", "odds_naive", 5, 14, 3.0),
    ]

    print(f"\n{'='*70}")
    print("  [5] 戦略別 × 券種別 利益")
    print(f"{'='*70}")

    for strat_name, score_name, pct, max_nh, max_fo in strategies:
        print("\n  ========================================")
        print(f"  戦略: {strat_name}")
        print("  ========================================")

        grand_tickets = {}
        for te_yr in TEST_YEARS:
            rf = race_data[te_yr]
            scores = year_scores[te_yr][score_name]

            mask = np.ones(len(rf), dtype=bool)
            mask &= (rf["n_horses"].values <= max_nh)
            mask &= (rf["fav1_odds"].values <= max_fo)

            filt_s = scores[mask]
            if len(filt_s) < 3:
                print(f"    {te_yr}: フィルタ後不足、スキップ")
                continue
            cutoff = np.percentile(filt_s, 100 - pct)
            sel_mask = mask & (scores >= cutoff)
            sel_rids = rf["race_id"].values[sel_mask].tolist()

            n_sel = len(sel_rids)
            # 荒れ的中数
            sel_upset = sum(
                int(rf[rf["race_id"] == rid]["actual_upset"].iloc[0])
                for rid in sel_rids
                if len(rf[rf["race_id"] == rid]) > 0
            )

            yr_tickets = compute_ticket_pnl(
                sel_rids,
                econ_data[te_yr]["hjc"],
                econ_data[te_yr]["nh"],
                econ_data[te_yr]["fav"],
            )

            print(f"\n    {te_yr}年: {n_sel}R選択,"
                  f" 荒れ的中: {sel_upset}/{n_sel}")

            print(f"    {'券種':>16s} {'購入額':>11s}"
                  f" {'払戻額':>11s} {'損益':>11s} {'ROI':>7s}")
            print("    " + "-" * 60)

            for tname, (c, p) in yr_tickets.items():
                roi = p / c * 100 if c > 0 else 0
                pnl = p - c
                print(f"    {tname:>16s} {c:>10,d}円"
                      f" {p:>10,d}円 {pnl:>+10,d}円 {roi:6.1f}%")

            # 累積
            for tname, (c, p) in yr_tickets.items():
                if tname not in grand_tickets:
                    grand_tickets[tname] = [0, 0]
                grand_tickets[tname][0] += c
                grand_tickets[tname][1] += p

        # 3年合計
        if grand_tickets:
            print("\n    --- 2024-2026 合計 ---")
            print(f"    {'券種':>16s} {'購入額':>11s}"
                  f" {'払戻額':>11s} {'損益':>11s} {'ROI':>7s}"
                  f" {'10枚利益':>13s}")
            print("    " + "-" * 73)
            for tname, (c, p) in grand_tickets.items():
                roi = p / c * 100 if c > 0 else 0
                pnl = p - c
                pnl10 = pnl * 10
                print(f"    {tname:>16s} {c:>10,d}円"
                      f" {p:>10,d}円 {pnl:>+10,d}円 {roi:6.1f}%"
                      f" {pnl10:>+12,d}円")

    elapsed = time.time() - start
    print(f"\n\n完了! ({elapsed:.1f}秒)")


if __name__ == "__main__":
    main()
