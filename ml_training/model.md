# 競馬荒れ予測モデル 技術仕様書

## 概要

JRDBデータを用いて「人気上位3頭（fav1, fav2, fav3）が全員負けるレース（=荒れるレース）」を予測し、
人気上位3頭を除外した馬券を購入することでROI>100%を目指すMLパイプライン。

2段階モデル（馬レベル予測→レースレベル予測）+ グリッドサーチによるROI最適化で構成される。

---

## 定義

| 定義 | 荒れの条件 | 馬モデルのターゲット | 主要券種 |
|------|-----------|-------------------|---------|
| **E定義** | fav1, fav2, fav3 が全員1着にこない | P(win) = P(着順==1) | 単勝ex3, 複勝ex3 |
| **F定義** | fav1, fav2, fav3 が全員2着以内にこない | P(top2) = P(着順<=2) | 馬単ex3, 三連単ex3 |

**ex3 = 人気上位3頭を除外して購入**
- 例: 14頭立てで単勝ex3 → 11頭分を各100円購入 = 1,100円/レース

---

## パイプライン全体像

```
[Phase 1] 馬レベル特徴量構築
    horse_model.build_horse_features(FG2-6, return_raw=True)
    → PostgreSQL: KYI⊲⊳BAC⊲⊳KAB⊲⊳SED + CYB/TYB/JOA/CHK/KKA/騎手/調教師/前走
    → 638,453行 × 198列 (1行 = 1出走馬)
        ↓
[Phase 2] 68ファクター計算
    evaluate_factors.compute_factors(df)
    → factor_definitions.py の77ファクター関数を適用
    → 0/1バイナリ列を68個追加
        ↓
[Phase 3] 7年ローリング学習 (フォールドごとに繰り返し)
    FOR each fold:
        ├─ L1正則化でファクター重み推定 → factor_score (1特徴量)
        ├─ LightGBM 7-seed ensemble → 馬ごとのP(win)またはP(top2)
        ├─ CatBoost 3-seed ensemble → 同上
        ├─ 55:45ブレンド + Isotonic Calibration
        └─ レースレベル集約 → _build_race_ef()
        ↓
[Phase 4] Stage2 荒れ予測モデル
    LightGBM (race-level features) → P(upset)
        ↓
[Phase 5] スコア計算 + グリッドサーチ
    compute_scores() → 20+種のスコア
    grid_search() → ~88,000通り探索
    → 全テスト年ROI>100%の戦略を抽出
```

---

## Phase 1: 馬レベル特徴量構築

### データソース (PostgreSQL)

| テーブル | 名称 | 主な列 |
|----------|------|--------|
| KYI | 前日_競走馬情報 | IDM, 騎手指数, 総合指数, 情報指数, 脚質, 印 |
| BAC | 前日_番組情報 | distance, surface(芝/ダ), grade, n_horses |
| KAB | 前日_開催情報 | weather, turf_condition, dirt_condition, bias |
| SED | 成績_成績分析用情報 | finish_pos, final_pop(確定人気), 着順 |
| CYB | 前日_調教分析情報 | workout_idx, fitness_idx, training_volume |
| TYB | 直前_情報 | paddock_idx, odds_idx, **day_win_odds** |
| JOA | 前日_詳細情報 | CID, LS指数, stable_bb_pr, jockey_bb_pr |
| CHK | 前日_調教本追切 | chk_workout_idx, chk_ten_idx, chk_end_idx |
| KKA | 前日_競走馬拡張 | surface_pr(連対率), sire_turf_rate, sire_dirt_rate |
| KZA/CZA | 騎手/調教師 | jockey_leading, jockey_wr, trainer_leading |
| SED(前走) | 前走1-5走 | prev1_finish, prev1_pop, prev1_agari, avg_finish_5 |
| HJC | 成績_払戻情報 | 単勝/複勝/馬連/馬単/三連複/三連単/ワイド払戻金 |

### 特徴量グループ (FG2-FG6)

| グループ | 内容 | 追加特徴量数 |
|----------|------|------------|
| FG2 | ペース/季節/枠 | 12 |
| FG3 | 騎手専門性（芝/ダ/距離） | 9 |
| FG4 | 馬体重（レース内偏差等） | 5 |
| FG5 | 重量種別+ローテーション | 6 |
| FG6 | コンボ+レース内集約 | 9 |

### レース内相対特徴量

```python
idm_rank_in_race    # IDMのレース内順位
odds_change         # day_win_odds - 基準オッズ
race_day_entropy    # オッズのエントロピー（混戦度）
is_top3_pop         # 人気上位3頭フラグ
```

### ターゲット変数

```python
win          = (finish_pos == 1)       # E定義用
top2_finish  = (finish_pos <= 2)       # F定義用
top3_finish  = (finish_pos <= 3)       # 未使用
```

### 最終特徴量

`build_horse_features(return_raw=True)` → 198列
- DROPする列: race_id, date, win, top2_finish, top3_finish, 生ID列, 印列, 個別ファクター列
- ML学習に使う列: **178特徴量**（factor_score含む、カテゴリ24列）

---

## Phase 2: 68ファクター

### ファクターとは

「穴馬シグナル」を検知するバイナリ指標。各馬に対して0/1で判定。

### カテゴリ別一覧

| カテゴリ | 個数 | 例 |
|----------|------|-----|
| A: JRDB指数乖離 | 12 | IDMレース内1位×人気薄, 荒れ印 |
| B: オッズ/市場乖離 | 7 | オッズ急落, 人気急上昇 |
| C: 出遅れ/不利 | 8 | 前走不利, 前走走力過小評価 |
| D: タイム/パフォーマンス | 7 | 上がり良好×着順悪い, トレンド上昇 |
| E: コース適性 | 7 | 芝/ダ適性, 種牡馬遺伝子率 |
| F: 脚質/ペース | 8 | 追込×ハイペース, 逃げ×スロー |
| G: 騎手/調教師 | 6 | リーディング騎手×人気薄, J×T組合せ |
| H: 調教 | 5 | 体力3位以内, 追切指数高い |
| I: 体重/ローテ/クラス | 8 | 体重増加, 格下げ, 季節 |

### factor_score の計算

```python
# フォールドごとにL1正則化ロジスティック回帰で重み推定
model = LogisticRegressionCV(
    Cs=20, cv=5, penalty='l1', solver='saga',
    scoring='roc_auc', max_iter=5000
)
# ターゲット: upset_horse = (final_pop > 3) & (finish_pos <= 3)
# 入力: 68ファクターの0/1列

# 非ゼロ重みのみ使用（典型的に40-50個が非ゼロ）
factor_score = Σ(factor_i × weight_i)
```

`factor_score` は178特徴量の1つとしてLGB/CBの学習に使用される。
個別の68ファクター列はDROPしてML学習には使わない。

---

## Phase 3: 馬モデル学習

### ローリングウィンドウ (7年固定)

| 学習期間 | テスト年 | L1学習 |
|----------|---------|--------|
| 2013-2019 | 2020 | 2013-2019 |
| 2014-2020 | 2021 | 2014-2020 |
| 2015-2021 | 2022 | 2015-2021 |
| 2016-2022 | 2023 | 2016-2022 |
| 2017-2023 | 2024 | 2017-2023 |
| 2018-2024 | 2025 | 2018-2024 |

### LightGBM (7-seed ensemble)

```python
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
# + "objective": "binary", "metric": "auc"
# + "scale_pos_weight": neg/pos (不均衡補正)
# num_boost_round=1200, early_stopping=50
# 7シードで学習し、median集約
```

AUC: 0.82-0.84

### CatBoost (3-seed ensemble)

```python
CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, eval_metric="AUC",
    early_stopping_rounds=50,
    cat_features=cat_indices,  # 24カテゴリ列
    scale_pos_weight=neg/pos,
)
# 3シードで学習し、median集約
```

AUC: 0.82-0.84

### ブレンド + Isotonic Calibration

```python
BLEND_WEIGHT_LGB = 0.55  # LGB 55% + CB 45%

# Isotonic Calibration:
# 学習データの最終1年をキャリブレーション用に分離
# cal_tr = train[:-1年], cal_te = train[-1年]
# キャリブレーション用にLGB/CBを別途学習
# IsotonicRegression(y_min=0.01, y_max=0.99) でブレンド出力を補正
# → テスト年の予測に適用
```

Calibrated AUC: 0.82-0.84

---

## Phase 3.5: レースレベル集約

### _build_race_ef() の処理

1. **fav1, fav2, fav3 の特定**: `day_win_odds` 昇順で上位3頭
2. **actual_upset ラベル**: SED確定人気で判定（E: 全員1着外, F: 全員2着外）
3. **レース特徴量を構築** (1行 = 1レース)

### レースレベル特徴量一覧

#### 馬モデル由来

| 特徴量 | 説明 |
|--------|------|
| p_fav1, p_fav2, p_fav3 | fav1-3のモデル予測確率 |
| p_product | (1-p1)×(1-p2)×(1-p3) — 荒れ確率のナイーブ推定 |
| p_product_odds | オッズ暗黙確率で同上 |
| fav1_odds, fav2_odds, fav3_odds | fav1-3のday_win_odds |
| fav_implied_sum | 1/odds1 + 1/odds2 + 1/odds3 |
| n_horses | 出走頭数 |
| p_rest_max, p_rest_mean | 非人気馬のモデル予測の最大/平均 |

#### 人気馬統計

| 特徴量 | 説明 |
|--------|------|
| h_fav_mean, h_fav_min, h_fav_max, h_fav_std | fav1-3予測の統計 |
| h_fav_logodds_sum | log-odds合計 |
| h_fav12_gap | p1 - p2（1番人気の突出度） |
| h_fav_weak_count | P < 0.5 の人気馬数 |
| h_fav_very_weak | P < 0.35 の人気馬数 |

#### 非人気馬統計

| 特徴量 | 説明 |
|--------|------|
| h_rest_std | 非人気馬予測のばらつき |
| h_rest_beat_fav3 | fav3より高予測の非人気馬数 |
| h_rest_top2_sum | 非人気馬上位2頭の予測合計 |
| h_rest_above_40, h_rest_above_50 | 高予測の非人気馬数 |

#### 全体統計

| 特徴量 | 説明 |
|--------|------|
| h_gap | fav平均 - rest平均 |
| h_all_std, h_all_range | 全馬予測のばらつき |
| h_pred_entropy | Shannon entropy（混戦度） |
| h_pred_gini | Gini不均衡度 |

#### ファクタースコア集約 (7特徴量)

| 特徴量 | 説明 |
|--------|------|
| fs_rest_max | 非人気馬のfactor_score最大値 |
| fs_rest_mean | 非人気馬のfactor_score平均 |
| fs_fav_mean | fav1-3のfactor_score平均 |
| fs_gap | rest_max - fav_mean（穴馬優位度） |
| fs_rest_top2_sum | 非人気馬上位2頭のfactor_score合計 |
| fs_rest_positive_n | factor_score > 0 の非人気馬数 |
| fs_rest_strong_n | factor_score > 0.3 の非人気馬数 |

---

## Phase 4: Stage2 荒れ予測モデル

Stage1のレースレベル特徴量を入力として、レース単位の荒れ確率を予測するLightGBMモデル。

### 学習方式

```python
# テスト年2022の場合:
#   Stage2学習データ = 2020年+2021年のレースレベル特徴量
#   テスト年2020は学習データなし → p_product にfallback

# LGB 10-seed ensemble (軽量モデル)
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 15,
    "max_depth": 4, "min_child_samples": 30,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "scale_pos_weight": neg/pos,
}
# num_boost_round=200 (軽量)
# 10シードのmedian集約
```

Stage2 AUC: 0.60-0.67（レースレベルなのでサンプル少 = AUC低め）

---

## Phase 5: スコア計算

### compute_scores() が生成するスコア

```python
# p = p_product (馬モデル由来)
# po = p_product_odds (オッズ由来)
# fis = fav_implied_sum
# fi1 = fav1_implied
# nh = n_horses
# prm = p_rest_max
# s2 = stage2_score
# fs_rm = fs_rest_max
# fs_gap = fs_gap

scores = {
    # 基本スコア
    "naive":        p,
    "odds_naive":   po,
    "blend50":      0.5*p + 0.5*po,
    "blend70":      0.7*p + 0.3*po,
    "payout_w":     p * fis,
    "payout_f1":    p * fi1,
    "field_adj":    p / sqrt(nh),
    "rest_threat":  p * prm,
    "combo":        p * fis * prm,
    "ev_excl":      p * fis / ((nh-3)*(nh-4)/2) * 100,

    # Stage2 スコア
    "s2_pure":      s2,
    "s2_blend50":   0.5*s2 + 0.5*p,
    "s2_blend70":   0.7*s2 + 0.3*p,
    "s2_x_odds":    s2 * fis,
    "s2_x_rest":    s2 * prm,

    # ファクター系スコア
    "factor_pure":     fs_rm,
    "p_x_factor":      p * max(fs_rm, 0.01),
    "factor_gap":      fs_gap,
    "combo_factor":    p * fis * max(fs_rm, 0.01),
    "s2_x_factor":     s2 * max(fs_rm, 0.01),
    "s2_blend_factor": 0.5*s2 + 0.3*p + 0.2*fs_rm,

    # リッチ特徴量
    "entropy_w":    p * entropy,
    "logodds_inv":  p * max(-logodds, 0),
}
```

---

## Phase 6: グリッドサーチ

### 探索空間

```python
score_names  = 20+種
pct_list     = [1, 2, 3, ..., 20, 25, 30]  # 上位何%を選択
ff_list      = [99, 16, 14, 12, 10]         # 出走頭数フィルタ
fav1_odds    = [99, 3.0, 2.5, 2.0, 1.5]     # fav1オッズフィルタ
ticket_names = [単勝ex3, 複勝ex3, 馬連ex3, 馬単ex3, 三連複ex3, 三連単ex3, ワイドex3]
# 合計: ~88,550通り
```

### 選択ロジック

```
各(score, pct, ff, fo, ticket)の組合せに対して:
  1. 各テスト年のレースをff, foでフィルタ
  2. フィルタ後スコアの上位pct%を超えるレースを選択
  3. 選択レースのROIを計算
  4. 全テスト年でROI>100% → winners
  5. N-1年以上ROI>100% → candidates
```

### 券種別コスト計算 (fav3除外)

| 券種 | 1レースのコスト (n頭立て) |
|------|-------------------------|
| 単勝ex3 | (n-3) × 100円 |
| 複勝ex3 | (n-3) × 100円 |
| 馬連ex3 | C(n-3, 2) × 100円 |
| 馬単ex3 | P(n-3, 2) × 100円 |
| 三連複ex3 | C(n-3, 3) × 100円 |
| 三連単ex3 | P(n-3, 3) × 100円 |
| ワイドex3 | C(n-3, 2) × 100円 |

例: 14頭立ての三連単ex3 = P(11,3) × 100 = 990 × 100 = 99,000円/レース

払い戻し計算時、fav1/fav2/fav3が含まれる組合せの払戻金は除外する。

---

## 最新結果 (2025年2月時点)

### テスト設定
- 7年スライディングウィンドウ
- テスト年: 2020, 2021, 2022, 2023, 2024, 2025
- L1重みはLGBと同じ学習期間を使用

### E定義: 6年全勝戦略 (3件)

| # | スコア | 選択 | フィルタ | 券種 | avg ROI | min ROI | 合計R |
|---|--------|------|---------|------|---------|---------|-------|
| 1 | naive | top6% | nh≤12 fo≤1.5 | 複勝ex3 | **141.2%** | **102.1%** | 60R |
| 2 | logodds_inv | top9% | nh≤12 fo≤1.5 | 複勝ex3 | 124.3% | 101.5% | 89R |
| 3 | logodds_inv | top2% | nh≤10 | 単勝ex3 | 108.3% | 100.8% | 68R |

### F定義: 6年全勝戦略なし

惜しい候補（5年ROI>100%）:
- s2_x_factor top3% fo≤1.5 単勝ex3（2023年のみ95.2%）
- payout_f1 top4% nh≤12 fo≤1.5 複勝ex3（2025年のみ98.4%）
- s2_x_odds top30% nh≤10 fo≤1.5 三連単ex3（2020年のみ49.6%）

### ベスト戦略詳細: naive top6% nh≤12 fo≤1.5 複勝ex3

```
年    ROI     レース  的中  的中率  投資       払戻       利益
2020  131.2%    8R    6h    75%    6,400円    8,400円    +2,000円
2021  191.2%   10R    8h    80%    7,600円   14,530円    +6,930円
2022  151.5%   11R   11h   100%    8,700円   13,180円    +4,480円
2023  140.1%   10R   10h   100%    7,100円    9,950円    +2,850円
2024  102.1%   12R   12h   100%    9,500円    9,700円      +200円
2025  130.8%    9R    9h   100%    7,300円    9,550円    +2,250円
合計  140.2%   60R   56h    93%   46,600円   65,310円   +18,710円
```

---

## ファイル構成

```
ml_training/
├── upset_ef.py              # メインパイプライン (全Phase統合)
├── horse_model.py           # Phase1: 馬レベル特徴量構築
├── factor_definitions.py    # Phase2: 68ファクター定義
├── evaluate_factors.py      # Phase2: ファクター評価・L1重み最適化
├── config.py                # DB接続設定・テーブル名定義
├── simulate_fixed20.py      # 固定枚数購入シミュレーション
├── simulate_holdings.py     # 所持金連動シミュレーション
├── show_strategy_detail.py  # 戦略別年別詳細表示
├── analyze_monthly.py       # 月別P&L分析
└── output/
    ├── E_detail.pkl          # E定義の全詳細データ
    └── F_detail.pkl          # F定義の全詳細データ
```

---

## 実行方法

```bash
cd ml_training
python upset_ef.py
# → 約3時間（LGB+CB有効時）
# → output/E_detail.pkl, F_detail.pkl を出力
```

### フラグ

| 変数 | 場所 | 説明 |
|------|------|------|
| `L1_ONLY` | upset_ef.py L49 | True: LGB/CBスキップ、False: フルパイプライン |
| `BEST_FEATURE_GROUPS` | upset_ef.py L38 | 使用する特徴量グループ |
| `BLEND_WEIGHT_LGB` | upset_ef.py L46 | LGB/CBのブレンド比率 |
| `N_ENS_CB` | upset_ef.py L45 | CatBoostのアンサンブル数 |
| `configs` | upset_ef.py main()内 | ローリングウィンドウ定義 |
