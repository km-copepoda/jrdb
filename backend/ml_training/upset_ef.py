"""
モデル評価パイプライン

処理フロー:
1. build_horse_features() -> prepare() (ラベルエンコーディング)
2. train(): LightGBM 7-seed ensemble (seed: 42, 49, 56, ... 84)
3. build_race_features(): レース単位で集計
   - 当日オッズ上位3頭を fav1, fav2, fav3 として抽出
   - p_threat: 4番人気以下の馬の勝率合計 sum(P(win))
   - fav_win_prob: fav1, 2, 3 が3着ともに入らない確率
   - upset_score: 実績 vs 予測の乖離 (荒れ度合い)
4. compute_scores():
   - naive: P_product
   - odds_naive: 1 / odds
   - blend50: 0.5 * naive + 0.5 * odds_naive
   - rest_threat: p_threat / sum(P_win)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import TUNED_PARAMS, BASE_PARAMS, SEEDS


def train(X_train, y_train, X_val, y_val):
    """LightGBM 7-seed ensemble で学習"""
    models = []
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    for seed in SEEDS:
        params = {
            **BASE_PARAMS,
            **TUNED_PARAMS,
            "seed": seed,
            "scale_pos_weight": neg / pos,
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
        models.append(model)
    return models


def predict_ensemble(models, X):
    """7モデルの平均予測"""
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1)


def build_race_features(df, pred_col="pred"):
    """
    レース単位で集計:
    - fav1, fav2, fav3: 当日オッズ上位3頭 (tyb_win_odds が小さい順)
    - p_threat: 4番人気以下の sum(P(win))
    - fav_win_prob: fav1-3 が全員 top3 に入らない確率
    - upset_score: 荒れ度合い
    """
    results = []

    for race_id, grp in df.groupby("race_id"):
        grp = grp.sort_values("tyb_win_odds", ascending=True).reset_index(drop=True)

        if len(grp) < 4:
            continue

        # fav1, fav2, fav3
        fav1_row = grp.iloc[0]
        fav2_row = grp.iloc[1]
        fav3_row = grp.iloc[2]

        # 人気上位3頭の予測勝率
        fav1_pred = fav1_row[pred_col]
        fav2_pred = fav2_row[pred_col]
        fav3_pred = fav3_row[pred_col]

        # 人気上位3頭が全員3着以内に入った (実績)
        fav1_top3 = fav1_row.get("top3_finish", np.nan)
        fav2_top3 = fav2_row.get("top3_finish", np.nan)
        fav3_top3 = fav3_row.get("top3_finish", np.nan)
        all_fav_in_top3 = int(fav1_top3 == 1 and fav2_top3 == 1 and fav3_top3 == 1)

        # p_threat: 4番人気以下の馬の P(win) 合計
        rest = grp.iloc[3:]
        p_threat = rest[pred_col].sum()

        # sum(P_win) 全馬
        total_p = grp[pred_col].sum()

        # fav_win_prob: fav1-3の予測top3確率の積 (全員がtop3に入る確率)
        fav_win_prob = fav1_pred * fav2_pred * fav3_pred

        # upset_score: 実績(荒れ=1) vs 予測(人気馬が入る確率) の乖離
        upset_actual = 1 - all_fav_in_top3  # 荒れた=1
        upset_score = upset_actual - fav_win_prob

        # 各馬の情報を保持
        race_info = {
            "race_id": race_id,
            "date": grp.iloc[0]["date"],
            "year": grp.iloc[0]["year"],
            "horse_count": len(grp),
            "fav1_umaban": fav1_row["umaban"],
            "fav2_umaban": fav2_row["umaban"],
            "fav3_umaban": fav3_row["umaban"],
            "fav1_pred": fav1_pred,
            "fav2_pred": fav2_pred,
            "fav3_pred": fav3_pred,
            "fav1_odds": fav1_row["tyb_win_odds"],
            "fav2_odds": fav2_row["tyb_win_odds"],
            "fav3_odds": fav3_row["tyb_win_odds"],
            "fav1_top3": fav1_top3,
            "fav2_top3": fav2_top3,
            "fav3_top3": fav3_top3,
            "all_fav_in_top3": all_fav_in_top3,
            "upset": upset_actual,
            "p_threat": p_threat,
            "total_p": total_p,
            "fav_win_prob": fav_win_prob,
            "upset_score": upset_score,
        }
        results.append(race_info)

    race_df = pd.DataFrame(results)
    print(f"build_race_features: {len(race_df)} races")
    return race_df


def compute_scores(race_df):
    """
    スコア計算:
    - naive: P_product (fav1-3 の予測確率の積)
    - odds_naive: 1 / (fav1_odds * fav2_odds * fav3_odds)
    - blend50: 0.5 * naive + 0.5 * odds_naive (正規化後)
    - rest_threat: p_threat / total_p
    """
    df = race_df.copy()

    # naive: 人気上位3頭の勝率積 (高いほど「堅い」)
    df["naive"] = df["fav1_pred"] * df["fav2_pred"] * df["fav3_pred"]

    # odds_naive: オッズベースの堅さ指標
    df["odds_naive"] = 1.0 / (df["fav1_odds"] * df["fav2_odds"] * df["fav3_odds"])

    # 正規化 (0-1 スケール)
    def _minmax(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return s * 0.0
        return (s - mn) / (mx - mn)

    naive_norm = _minmax(df["naive"])
    odds_norm = _minmax(df["odds_naive"])

    # blend50: モデル予測とオッズのブレンド
    df["blend50"] = 0.5 * naive_norm + 0.5 * odds_norm

    # rest_threat: 穴馬脅威度 (高いほど荒れそう)
    df["rest_threat"] = df["p_threat"] / df["total_p"].replace(0, np.nan)

    print(f"compute_scores: 荒れ率={df['upset'].mean():.1%}")
    return df


def evaluate_upset_prediction(scored_df, score_col="rest_threat", top_pct=0.03):
    """
    荒れ予測の精度評価
    score_col の上位 top_pct% を荒れレースと予測し、実際の荒れ率と比較
    """
    df = scored_df.copy()
    threshold = df[score_col].quantile(1 - top_pct)
    predicted_upset = df[df[score_col] >= threshold]

    actual_upset_rate = predicted_upset["upset"].mean()
    baseline_rate = df["upset"].mean()

    print(f"\n=== 荒れ予測評価 ({score_col} top {top_pct:.0%}) ===")
    print(f"全体の荒れ率:      {baseline_rate:.1%}")
    print(f"予測上位の荒れ率:  {actual_upset_rate:.1%}")
    print(f"リフト:            {actual_upset_rate / baseline_rate:.2f}x")
    print(f"対象レース数:      {len(predicted_upset)}")

    return {
        "score_col": score_col,
        "top_pct": top_pct,
        "baseline_rate": baseline_rate,
        "predicted_rate": actual_upset_rate,
        "lift": actual_upset_rate / baseline_rate if baseline_rate > 0 else 0,
        "n_races": len(predicted_upset),
    }


if __name__ == "__main__":
    from horse_model import build_horse_features, prepare, FEATURE_COLS

    print("=== upset_ef.py: モデル評価パイプライン ===")

    # Step 1: 特徴量生成
    print("\nStep 1: 特徴量生成...")
    df = build_horse_features()

    # Step 2: 前処理
    print("\nStep 2: 前処理...")
    df, features, _ = prepare(df)

    # Step 3: ローリング学習・予測
    print("\nStep 3: ローリング学習・予測...")
    all_test = []

    for year in [2024, 2025, 2026]:
        train_mask = df["year"] < year
        test_mask = df["year"] == year

        if test_mask.sum() == 0:
            print(f"  {year}: テストデータなし → スキップ")
            continue

        train_df = df[train_mask]
        test_df = df[test_mask].copy()

        # 直近1年をバリデーション
        val_year = year - 1
        val_mask = train_df["year"] == val_year
        if val_mask.sum() == 0:
            val_mask = train_df["year"] == train_df["year"].max()

        X_train = train_df[~val_mask][features]
        y_train = train_df[~val_mask]["win"]
        X_val = train_df[val_mask][features]
        y_val = train_df[val_mask]["win"]
        X_test = test_df[features]

        print(f"\n  {year}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        models = train(X_train, y_train, X_val, y_val)
        test_df["pred"] = predict_ensemble(models, X_test)
        all_test.append(test_df)

    if not all_test:
        print("テストデータがありません。")
    else:
        combined = pd.concat(all_test, ignore_index=True)

        # Step 4: レース特徴量
        print("\nStep 4: レース特徴量集計...")
        race_df = build_race_features(combined)

        # Step 5: スコア計算
        print("\nStep 5: スコア計算...")
        scored = compute_scores(race_df)

        # Step 6: 評価
        print("\nStep 6: 荒れ予測評価...")
        for col in ["naive", "blend50", "rest_threat"]:
            # rest_threat は高いほど荒れやすい、naive/blend50は低いほど荒れやすい
            if col == "rest_threat":
                evaluate_upset_prediction(scored, col, top_pct=0.03)
            else:
                # naive, blend50 は逆: 下位 = 荒れやすい → 1-score で反転
                scored_inv = scored.copy()
                scored_inv[f"{col}_inv"] = 1 - scored_inv[col]
                evaluate_upset_prediction(scored_inv, f"{col}_inv", top_pct=0.03)

        # 結果保存
        scored.to_csv("upset_scores.csv", index=False)
        print("\n結果を upset_scores.csv に保存しました。")
