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
import lightgbm as lgb

from config import TUNED_PARAMS, BASE_PARAMS, SEEDS
from horse_model import build_horse_features, prepare


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
    - fav1, fav2, fav3: 当日オッズ上位3頭
    - p_threat: 4番人気以下の sum(P(win))
    - fav_win_prob: fav1-3 が全員 top3 に入らない確率
    - upset_score: 荒れ度合い
    """
    # TODO: 実装
    raise NotImplementedError("build_race_features() は未実装です")


def compute_scores(race_df):
    """
    スコア計算:
    - naive: P_product
    - odds_naive: 1 / odds
    - blend50: 0.5 * naive + 0.5 * odds_naive
    - rest_threat: p_threat / sum(P_win)
    """
    # TODO: 実装
    raise NotImplementedError("compute_scores() は未実装です")


if __name__ == "__main__":
    print("upset_ef.py: モデル評価パイプライン")
    print("使い方: python upset_ef.py")
    # TODO: メインの実行ロジック
