"""
穴馬スコアリングファクター評価スクリプト

83個のファクターの予測力を統計的に評価し、最適重みを決定する。

出力:
  - コンソール: ファクター別メトリクス、カテゴリ別サマリー、最適重み、年別安定性
  - CSV: output/factor_report.csv, output/factor_weights.csv, output/factor_yearly.csv

使い方:
  cd ml_training && python evaluate_factors.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
import lightgbm as lgb

from horse_model import build_horse_features, _to_numeric
from config import DB_CONFIG, T_KYI, T_SED, T_HJC
from factor_definitions import FACTOR_CATALOG, ALL_FACTOR_NAMES, CATEGORIES

warnings.filterwarnings("ignore")

BEST_FG = ["FG2", "FG3", "FG4", "FG5", "FG6"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TEST_YEARS = ["2023", "2024", "2025"]
TRAIN_END = "2022"


# ================================================================
# 1. 追加データの取得
# ================================================================
def _load_extra_prev_cols():
    """SED prev1/prev2 から不利・タイム等の追加列を取得"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        sql = f"""
        SELECT
            kyi."競走馬情報ID" AS horse_race_id,
            prev1."不利"           AS prev1_furi,
            prev1."前不利"         AS prev1_mae_furi,
            prev1."中不利"         AS prev1_naka_furi,
            prev1."後不利"         AS prev1_ato_furi,
            prev1."レース"         AS prev1_race_eval,
            prev1."素点"           AS prev1_soten,
            prev1."前3Fタイム"     AS prev1_mae3f,
            prev1."後3Fタイム"     AS prev1_ato3f,
            prev1."斤量"           AS prev1_kinryo,
            prev1."距離"           AS prev1_distance,
            prev1."芝ダ障害コード" AS prev1_surface,
            prev1."グレードコード" AS prev1_grade,
            prev1."馬体重"         AS prev1_weight,
            prev1."馬体重増減"     AS prev1_weight_change,
            prev2."不利"           AS prev2_furi
        FROM "{T_KYI}" kyi
        LEFT JOIN "{T_SED}" prev1
            ON prev1."馬基本情報_id" = SUBSTRING(kyi."前走1競走成績キー" FROM 1 FOR 8)
           AND TRIM(prev1."年月日") = SUBSTRING(kyi."前走1競走成績キー" FROM 9 FOR 8)
        LEFT JOIN "{T_SED}" prev2
            ON prev2."馬基本情報_id" = SUBSTRING(kyi."前走2競走成績キー" FROM 1 FOR 8)
           AND TRIM(prev2."年月日") = SUBSTRING(kyi."前走2競走成績キー" FROM 9 FOR 8)
        """
        extra = pd.read_sql(sql, conn)
        num_cols = ["prev1_furi", "prev1_mae_furi", "prev1_naka_furi",
                    "prev1_ato_furi", "prev1_race_eval", "prev1_soten",
                    "prev1_mae3f", "prev1_ato3f", "prev1_kinryo",
                    "prev1_distance", "prev1_grade", "prev1_weight",
                    "prev1_weight_change", "prev2_furi"]
        for col in num_cols:
            if col in extra.columns:
                extra[col] = _to_numeric(extra[col])
        return extra
    finally:
        conn.close()


def _load_payouts(race_ids):
    """単勝・複勝の払戻情報を取得"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ids_sql = ", ".join(f"'{r}'" for r in race_ids)
        cols = []
        for i in range(1, 4):
            cols += [f"単勝払戻{i}_払戻金", f"単勝払戻{i}_馬番"]
        for i in range(1, 6):
            cols += [f"複勝払戻{i}_払戻金", f"複勝払戻{i}_馬番"]
        cols_sql = ", ".join(f'hjc."{c}"' for c in cols)
        sql = f"""
        SELECT hjc."前日_番組情報_id" AS race_id, {cols_sql}
        FROM "{T_HJC}" hjc
        WHERE hjc."前日_番組情報_id" IN ({ids_sql})
        """
        hjc = pd.read_sql(sql, conn)
        for c in cols:
            if "払戻金" in c:
                hjc[c] = pd.to_numeric(
                    hjc[c].astype(str).str.strip(), errors="coerce"
                ).fillna(0)
            else:
                hjc[c] = hjc[c].astype(str).str.strip()
        return hjc
    finally:
        conn.close()


# ================================================================
# 2. ファクター計算
# ================================================================
def compute_factors(df):
    """全ファクターを計算してDataFrameに追加"""
    factor_cols = []
    for name in ALL_FACTOR_NAMES:
        meta = FACTOR_CATALOG[name]
        fn = meta["fn"]
        try:
            col = fn(df)
            df[name] = col.values if hasattr(col, 'values') else col
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
            df[name] = 0
        factor_cols.append(name)
    return factor_cols


# ================================================================
# 3. 評価メトリクス
# ================================================================
def _cohens_d(group1, group2):
    """Cohen's d 効果量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(), group2.std()
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def _safe_auc(y_true, y_pred):
    """安全なAUC計算"""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def _chi2_p(factor, target):
    """カイ二乗検定のp値"""
    ct = pd.crosstab(factor, target)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 1.0
    try:
        _, p, _, _ = chi2_contingency(ct)
        return p
    except ValueError:
        return 1.0


def _build_payout_maps(hjc_df):
    """払戻データを (race_id, 馬番) → 払戻金 の辞書に変換 (1回だけ実行)"""
    tan_map = {}  # (race_id, horse_no_str) -> payout
    fuku_map = {}
    for _, row in hjc_df.iterrows():
        rid = row["race_id"]
        for i in range(1, 4):
            umaban = str(row.get(f"単勝払戻{i}_馬番", "")).strip()
            payout = row.get(f"単勝払戻{i}_払戻金", 0)
            if umaban and umaban != "nan" and payout > 0:
                tan_map[(rid, umaban)] = tan_map.get((rid, umaban), 0) + payout
        for i in range(1, 6):
            umaban = str(row.get(f"複勝払戻{i}_馬番", "")).strip()
            payout = row.get(f"複勝払戻{i}_払戻金", 0)
            if umaban and umaban != "nan" and payout > 0:
                fuku_map[(rid, umaban)] = fuku_map.get((rid, umaban), 0) + payout
    return tan_map, fuku_map


def compute_roi(df, factor_col, tan_map, fuku_map):
    """ファクター発火馬に単勝/複勝100円ずつ賭けた場合のROI (ベクトル化)"""
    triggered = df[df[factor_col] == 1]
    n = len(triggered)
    if n == 0:
        return np.nan, np.nan

    total_bet = n * 100

    # 馬番文字列を作成
    hno_str = triggered["horse_no"].dropna().astype(int).astype(str)
    keys = list(zip(triggered.loc[hno_str.index, "race_id"], hno_str))

    tansho_payout = sum(tan_map.get(k, 0) for k in keys)
    fukusho_payout = sum(fuku_map.get(k, 0) for k in keys)

    roi_tan = (tansho_payout / total_bet * 100) if total_bet > 0 else np.nan
    roi_fuku = (fukusho_payout / total_bet * 100) if total_bet > 0 else np.nan
    return roi_tan, roi_fuku


def evaluate_single_factor(df, factor_col, target_upset, target_outperform,
                           tan_map=None, fuku_map=None):
    """単一ファクターの全メトリクスを計算"""
    f = df[factor_col]
    valid = f.notna()
    f = f[valid].astype(int)
    y_upset = target_upset[valid]
    y_outperf = target_outperform[valid]

    coverage = f.mean()
    n_triggered = int(f.sum())

    if n_triggered == 0 or coverage >= 0.95:
        return {
            "factor": factor_col,
            "coverage": coverage,
            "n_triggered": n_triggered,
            "precision_lift": np.nan,
            "mean_outperform": np.nan,
            "effect_d": np.nan,
            "auc_upset": np.nan,
            "roi_tansho": np.nan,
            "roi_fukusho": np.nan,
            "chi2_p": np.nan,
        }

    # Precision lift
    base_rate = y_upset.mean()
    fired_rate = y_upset[f == 1].mean() if n_triggered > 0 else 0
    lift = (fired_rate / base_rate) if base_rate > 0 else np.nan

    # 発火時の平均 outperform
    mean_outperf = y_outperf[f == 1].mean()

    # Cohen's d
    d = _cohens_d(y_outperf[f == 1], y_outperf[f == 0])

    # AUC
    auc = _safe_auc(y_upset, f)

    # Chi2 p-value
    p = _chi2_p(f, y_upset)

    # ROI
    roi_tan, roi_fuku = np.nan, np.nan
    if tan_map is not None and n_triggered > 0:
        roi_tan, roi_fuku = compute_roi(df[valid], factor_col, tan_map, fuku_map)

    return {
        "factor": factor_col,
        "coverage": coverage,
        "n_triggered": n_triggered,
        "precision_lift": lift,
        "mean_outperform": mean_outperf,
        "effect_d": d,
        "auc_upset": auc,
        "roi_tansho": roi_tan,
        "roi_fukusho": roi_fuku,
        "chi2_p": p,
    }


# ================================================================
# 4. 年別安定性
# ================================================================
def evaluate_yearly(df, factor_cols, target_upset):
    """年別のAUCを計算"""
    df_year = df["date"].str[:4]
    results = []
    for fname in factor_cols:
        row = {"factor": fname}
        aucs = []
        for year in TEST_YEARS:
            mask = df_year == year
            if mask.sum() < 100:
                row[f"auc_{year}"] = np.nan
                continue
            auc = _safe_auc(target_upset[mask], df[fname][mask])
            row[f"auc_{year}"] = auc
            if not np.isnan(auc):
                aucs.append(auc)
        row["auc_std"] = np.std(aucs) if len(aucs) >= 2 else np.nan
        row["stable"] = row["auc_std"] < 0.02 if not np.isnan(row.get("auc_std", np.nan)) else False
        results.append(row)
    return pd.DataFrame(results)


# ================================================================
# 5. 重み最適化
# ================================================================
def optimize_weights(df, factor_cols, target):
    """L1正則化ロジスティック回帰で最適重みを算出"""
    X = df[factor_cols].fillna(0).astype(float)
    y = target.values

    # NaNを含む行を除外
    valid = ~(X.isna().any(axis=1) | pd.isna(y))
    X = X[valid]
    y = y[valid]

    if len(y) < 100 or y.sum() < 10:
        return pd.DataFrame()

    model = LogisticRegressionCV(
        Cs=20, cv=5, penalty='l1', solver='saga',
        scoring='roc_auc', max_iter=5000, random_state=42,
    )
    model.fit(X, y)

    weights = pd.DataFrame({
        "factor": factor_cols,
        "lr_weight": model.coef_[0],
        "lr_abs_weight": np.abs(model.coef_[0]),
    }).sort_values("lr_abs_weight", ascending=False)

    # 方向チェック
    for _, row in weights.iterrows():
        expected = FACTOR_CATALOG[row["factor"]]["dir"]
        actual_dir = 1 if row["lr_weight"] > 0 else (-1 if row["lr_weight"] < 0 else 0)
        weights.loc[weights["factor"] == row["factor"], "direction_ok"] = (
            "Yes" if actual_dir == expected else
            ("Zero" if actual_dir == 0 else "REVERSED")
        )

    return weights


def lgb_importance(df, factor_cols, target):
    """LightGBMで特徴量重要度を取得"""
    X = df[factor_cols].fillna(0).astype(float)
    y = target.values
    valid = ~pd.isna(y)
    X, y = X[valid], y[valid]

    if len(y) < 100:
        return pd.DataFrame()

    ds = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary", "metric": "auc",
        "num_leaves": 31, "max_depth": 4,
        "learning_rate": 0.1, "verbose": -1,
        "n_jobs": -1, "seed": 42,
    }
    model = lgb.train(params, ds, num_boost_round=200)
    imp = model.feature_importance(importance_type="gain")

    return pd.DataFrame({
        "factor": factor_cols,
        "lgb_gain": imp,
    }).sort_values("lgb_gain", ascending=False)


# ================================================================
# 6. レポート出力
# ================================================================
def print_report(results_df, yearly_df, weights_df, lgb_imp_df, cat_summary):
    """コンソールにレポートを出力"""
    print("\n" + "=" * 80)
    print("  穴馬ファクター評価レポート")
    print("=" * 80)

    # [1] Top 20 by Lift
    print("\n[1] Top 20 ファクター (Lift順)")
    print("-" * 100)
    top20 = results_df.dropna(subset=["precision_lift"]).sort_values(
        "precision_lift", ascending=False
    ).head(20)
    print(f"{'#':>3}  {'Factor':<30}  {'Coverage':>8}  {'Lift':>6}  {'AUC':>6}  "
          f"{'d':>6}  {'ROI(単)':>8}  {'ROI(複)':>8}  {'p値':>8}")
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        sig = "***" if r["chi2_p"] < 0.001 else "**" if r["chi2_p"] < 0.01 else "*" if r["chi2_p"] < 0.05 else ""
        desc = FACTOR_CATALOG[r["factor"]]["desc"]
        print(f"{i:>3}  {desc:<30}  {r['coverage']:>7.1%}  {r['precision_lift']:>6.2f}  "
              f"{r['auc_upset']:>6.3f}  {r['effect_d']:>6.3f}  "
              f"{r['roi_tansho']:>7.1f}%  {r['roi_fukusho']:>7.1f}%  {sig:>4}")

    # [2] カテゴリ別サマリー
    print(f"\n[2] カテゴリ別サマリー")
    print("-" * 80)
    print(f"{'Category':<20}  {'n':>3}  {'Avg_AUC':>8}  {'Best_AUC':>9}  "
          f"{'Avg_Lift':>9}  {'有効率':>6}")
    for _, r in cat_summary.iterrows():
        print(f"{r['category']:<20}  {r['n_factors']:>3}  {r['avg_auc']:>8.3f}  "
              f"{r['best_auc']:>9.3f}  {r['avg_lift']:>9.2f}  "
              f"{r['effective_rate']:>5.0%}")

    # [3] 最適重み
    if len(weights_df) > 0:
        print(f"\n[3] 最適重み (L1ロジスティック回帰, 上位20)")
        print("-" * 70)
        print(f"{'Factor':<30}  {'Original':>8}  {'LR_Weight':>10}  {'方向':>10}")
        for _, r in weights_df.head(20).iterrows():
            orig = FACTOR_CATALOG[r["factor"]]["dir"]
            desc = FACTOR_CATALOG[r["factor"]]["desc"]
            ok = r.get("direction_ok", "?")
            mark = "OK" if ok == "Yes" else ("ZERO" if ok == "Zero" else "NG!")
            print(f"{desc:<30}  {orig:>+8d}  {r['lr_weight']:>+10.3f}  {mark:>10}")

    # [4] LightGBM重要度
    if len(lgb_imp_df) > 0:
        print(f"\n[4] LightGBM Feature Importance (上位20)")
        print("-" * 50)
        for i, (_, r) in enumerate(lgb_imp_df.head(20).iterrows(), 1):
            desc = FACTOR_CATALOG[r["factor"]]["desc"]
            print(f"  {i:>2}. {desc:<35} gain={r['lgb_gain']:>8.1f}")

    # [5] 年別安定性
    print(f"\n[5] 年別安定性 (AUC > 0.52 のファクター)")
    print("-" * 80)
    stable = yearly_df[yearly_df[[f"auc_{y}" for y in TEST_YEARS]].mean(axis=1) > 0.52]
    if len(stable) > 0:
        print(f"{'Factor':<30}", end="")
        for y in TEST_YEARS:
            print(f"  {y:>6}", end="")
        print(f"  {'Std':>6}  {'Stable':>6}")
        for _, r in stable.sort_values("auc_std").head(20).iterrows():
            desc = FACTOR_CATALOG[r["factor"]]["desc"]
            print(f"{desc:<30}", end="")
            for y in TEST_YEARS:
                v = r.get(f"auc_{y}", np.nan)
                print(f"  {v:>6.3f}" if not np.isnan(v) else f"  {'N/A':>6}", end="")
            std_v = r.get("auc_std", np.nan)
            stbl = "Yes" if r.get("stable", False) else "No"
            print(f"  {std_v:>6.3f}  {stbl:>6}" if not np.isnan(std_v) else f"  {'N/A':>6}  {stbl:>6}")

    # [6] 方向が逆のファクター
    if len(weights_df) > 0:
        reversed_facs = weights_df[weights_df["direction_ok"] == "REVERSED"]
        if len(reversed_facs) > 0:
            print(f"\n[6] 方向が逆のファクター (想定と逆効果)")
            print("-" * 60)
            for _, r in reversed_facs.iterrows():
                desc = FACTOR_CATALOG[r["factor"]]["desc"]
                print(f"  {desc:<35}  expected={FACTOR_CATALOG[r['factor']]['dir']:+d}  "
                      f"actual_weight={r['lr_weight']:+.3f}")


# ================================================================
# 7. メイン
# ================================================================
def main():
    print("=" * 60)
    print("穴馬ファクター評価 - 開始")
    print("=" * 60)

    # --- Phase 1: データ読み込み ---
    print("\n[Phase 1] ベースデータ読み込み...")
    df = build_horse_features(feature_groups=BEST_FG, return_raw=True)
    print(f"  ベース: {df.shape[0]} 行 x {df.shape[1]} 列")

    # --- Phase 2: 追加SED列 ---
    print("\n[Phase 2] 追加SED列(不利/タイム等)を取得中...")
    extra = _load_extra_prev_cols()
    # horse_race_id でマージ (重複列を避ける)
    existing_cols = set(df.columns)
    extra_new_cols = [c for c in extra.columns
                      if c not in existing_cols or c == "horse_race_id"]
    df = df.merge(extra[extra_new_cols], on="horse_race_id", how="left")
    print(f"  追加後: {df.shape[0]} 行 x {df.shape[1]} 列")

    # --- Phase 3: ターゲット変数 ---
    print("\n[Phase 3] ターゲット変数を計算中...")
    df["finish_pos"] = _to_numeric(df["finish_pos"])
    df["final_pop"] = _to_numeric(df["final_pop"])

    # 異常値を除外 (取消馬等)
    valid_mask = df["finish_pos"].notna() & df["final_pop"].notna()
    df = df[valid_mask].copy()

    df["outperform_odds"] = df["final_pop"] - df["finish_pos"]
    df["upset_horse"] = ((df["final_pop"] > 3) & (df["finish_pos"] <= 3)).astype(int)
    df["hidden_quality"] = (df["outperform_odds"] >= 3).astype(int)

    print(f"  有効データ: {len(df)} 行")
    print(f"  upset_horse 発生率: {df['upset_horse'].mean():.1%}")
    print(f"  hidden_quality 発生率: {df['hidden_quality'].mean():.1%}")

    # --- Phase 4: ファクター計算 ---
    print("\n[Phase 4] 83ファクターを計算中...")
    factor_cols = compute_factors(df)
    print(f"  計算完了: {len(factor_cols)} ファクター")

    # 各ファクターの発火数を簡易表示
    n_triggered = {f: int(df[f].sum()) for f in factor_cols}
    zero_factors = [f for f, n in n_triggered.items() if n == 0]
    if zero_factors:
        print(f"  [注意] 発火数0のファクター: {len(zero_factors)}個")
        for f in zero_factors:
            print(f"    - {FACTOR_CATALOG[f]['desc']}")

    # --- Phase 5: 払戻データ取得 ---
    print("\n[Phase 5] 払戻データを取得中...")
    # テスト年のみROIを計算 (全期間は遅すぎる)
    test_mask = df["date"].str[:4].isin(TEST_YEARS)
    test_race_ids = df[test_mask]["race_id"].unique().tolist()
    hjc_df = _load_payouts(test_race_ids)
    print(f"  テスト期間レース数: {len(test_race_ids)}")

    # 払戻マップを事前構築 (高速化)
    print("  払戻マップを構築中...")
    tan_map, fuku_map = _build_payout_maps(hjc_df)
    print(f"  単勝マップ: {len(tan_map)} 件, 複勝マップ: {len(fuku_map)} 件")

    # --- Phase 6: ファクター評価 ---
    print("\n[Phase 6] ファクター評価中...")
    test_df = df[test_mask].copy().reset_index(drop=True)
    target_upset = test_df["upset_horse"]
    target_outperform = test_df["outperform_odds"]

    results = []
    for i, fname in enumerate(factor_cols, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(factor_cols)}...")
        r = evaluate_single_factor(
            test_df, fname, target_upset, target_outperform, tan_map, fuku_map
        )
        # メタデータ追加
        r["category"] = FACTOR_CATALOG[fname]["cat"]
        r["description"] = FACTOR_CATALOG[fname]["desc"]
        r["expected_dir"] = FACTOR_CATALOG[fname]["dir"]
        results.append(r)

    results_df = pd.DataFrame(results)

    # --- Phase 7: 年別安定性 ---
    print("\n[Phase 7] 年別安定性を評価中...")
    yearly_df = evaluate_yearly(df, factor_cols, df["upset_horse"])

    # --- Phase 8: 重み最適化 ---
    print("\n[Phase 8] 重み最適化中...")
    # 訓練データ (テスト年より前)
    train_mask = df["date"].str[:4] <= TRAIN_END
    train_df = df[train_mask].copy().reset_index(drop=True)

    weights_df = optimize_weights(
        train_df, factor_cols, train_df["upset_horse"]
    )
    lgb_imp_df = lgb_importance(
        train_df, factor_cols, train_df["upset_horse"]
    )

    # --- Phase 9: カテゴリ別サマリー ---
    cat_rows = []
    for cat, factors in sorted(CATEGORIES.items()):
        cat_results = results_df[results_df["category"] == cat]
        aucs = cat_results["auc_upset"].dropna()
        lifts = cat_results["precision_lift"].dropna()
        cat_rows.append({
            "category": cat,
            "n_factors": len(factors),
            "avg_auc": aucs.mean() if len(aucs) > 0 else np.nan,
            "best_auc": aucs.max() if len(aucs) > 0 else np.nan,
            "avg_lift": lifts.mean() if len(lifts) > 0 else np.nan,
            "effective_rate": (aucs > 0.52).mean() if len(aucs) > 0 else 0,
        })
    cat_summary = pd.DataFrame(cat_rows).sort_values("avg_auc", ascending=False)

    # --- Phase 10: レポート出力 ---
    print_report(results_df, yearly_df, weights_df, lgb_imp_df, cat_summary)

    # CSV保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "factor_report.csv"), index=False)
    yearly_df.to_csv(os.path.join(OUTPUT_DIR, "factor_yearly.csv"), index=False)
    if len(weights_df) > 0:
        weights_df.to_csv(os.path.join(OUTPUT_DIR, "factor_weights.csv"), index=False)
    if len(lgb_imp_df) > 0:
        lgb_imp_df.to_csv(os.path.join(OUTPUT_DIR, "factor_lgb_importance.csv"), index=False)
    cat_summary.to_csv(os.path.join(OUTPUT_DIR, "factor_category_summary.csv"), index=False)

    print(f"\n結果を {OUTPUT_DIR}/ に保存しました")
    print("完了!")


if __name__ == "__main__":
    main()
