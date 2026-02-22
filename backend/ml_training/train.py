"""
特徴量生成・学習の実行スクリプト

使い方:
    cd ml_training
    python train.py

処理:
1. build_horse_features() で特徴量 DataFrame を構築
2. prepare() でラベルエンコーディング
3. ローリング評価: 2018-N-1年で学習 -> N+1年を予測
4. LightGBM 7-seed ensemble で学習
"""

import pickle
import pandas as pd
from horse_model import build_horse_features, prepare, FEATURE_COLS
from upset_ef import train, predict_ensemble


def main():
    # Step 1: 特徴量生成
    print("Step 1: 特徴量生成...")
    df = build_horse_features()

    # Step 2: 前処理
    print("Step 2: 前処理...")
    df, features, label_encoders = prepare(df)

    # Step 3: ローリング学習・予測
    print("Step 3: ローリング学習・予測...")
    all_test = []
    all_models = {}

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
        all_models[year] = models

        print(f"  {year}: 学習完了 (7-seed ensemble)")

    # 結果保存
    if all_test:
        combined = pd.concat(all_test, ignore_index=True)
        combined.to_csv("predictions.csv", index=False)
        print(f"\n予測結果を predictions.csv に保存 ({len(combined)} rows)")

        # モデル保存
        with open("models.pkl", "wb") as f:
            pickle.dump(all_models, f)
        print("モデルを models.pkl に保存")

        # ラベルエンコーダー保存
        with open("label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        print("ラベルエンコーダーを label_encoders.pkl に保存")
    else:
        print("\nテストデータがありません。")

    print("\n完了")


if __name__ == "__main__":
    main()
