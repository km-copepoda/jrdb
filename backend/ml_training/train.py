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

from horse_model import build_horse_features, prepare
from upset_ef import train, predict_ensemble


def main():
    # TODO: 特徴量生成
    print("Step 1: 特徴量生成...")
    # df = build_horse_features()

    # TODO: 前処理
    print("Step 2: 前処理...")
    # df = prepare(df)

    # TODO: ローリング学習・予測
    print("Step 3: ローリング学習・予測...")
    # for year in [2024, 2025, 2026]:
    #     train_df = df[df['year'] < year]
    #     test_df = df[df['year'] == year]
    #     models = train(X_train, y_train, X_val, y_val)
    #     preds = predict_ensemble(models, X_test)

    print("完了")


if __name__ == "__main__":
    main()
