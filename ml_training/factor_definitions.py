"""
穴馬スコアリングファクター定義 v2 (77個, 9カテゴリ)

v1 → v2 変更点:
  - データ形式修正: pace_pred H/M/S, 激走印>=1, grade空文字列
  - 削除 11個: 空データ(BB率), 逆効果で冗長なもの
  - 方向反転 5個: 「好調=織り込み済み」→ dir=-1 (過大評価シグナル)
  - 新規追加 5個: 代替ファクター

dir=+1: 穴馬シグナル (発火=過小評価の可能性)
dir=-1: 人気馬過大評価シグナル (発火=期待ほどではない可能性)
"""
import numpy as np
import pandas as pd


# ================================================================
# ヘルパー
# ================================================================
def _unpopular(df, threshold=4):
    """人気薄 = 基準人気順位が threshold 以上"""
    return df["base_pop_rank"] >= threshold


def _rank_in_race(df, col, ascending=False):
    """レース内でのランクを計算"""
    return df.groupby("race_id")[col].rank(
        ascending=ascending, method="min"
    )


# ================================================================
# カテゴリ A: JRDB指数と人気の乖離 (12個)
# ================================================================
def f336(df):
    """IDMレース内1位なのに人気4番以下"""
    return ((df["idm_rank_in_race"] == 1) & _unpopular(df)).astype(int)

def f337(df):
    """IDMレース内上位25%なのに人気薄"""
    q = df["n_horses"] / 4.0
    return ((df["idm_rank_in_race"] <= q) & _unpopular(df, 5)).astype(int)

def f345(df):
    """CIDレース内上位3なのに人気薄"""
    rank = _rank_in_race(df, "cid")
    return ((rank <= 3) & _unpopular(df)).astype(int)

def f347(df):
    """上がり指数順位がメンバー中1位なのに人気薄"""
    return ((df["agari_rank"] == 1) & _unpopular(df)).astype(int)

def f348(df):
    """テン指数順位がメンバー中1位なのに人気薄"""
    return ((df["ten_rank"] == 1) & _unpopular(df)).astype(int)

def f349(df):
    """ペース指数レース内1位なのに人気薄"""
    rank = _rank_in_race(df, "ペース指数")
    return ((rank == 1) & _unpopular(df)).astype(int)

def f240(df):
    """IDM印が○以上なのに人気5番以下"""
    return ((df["mark_idm_num"] >= 2) & _unpopular(df, 5)).astype(int)

def f241(df):
    """総合印が○以上なのに人気5番以下"""
    return ((df["mark_overall_num"] >= 2) & _unpopular(df, 5)).astype(int)

def f243(df):
    """激走印あり (>=1)"""
    return (df["mark_upset_num"].fillna(0) >= 1).astype(int)

def f244(df):
    """万券印あり (>=1)"""
    return (df["mark_longshot_num"].fillna(0) >= 1).astype(int)

def f251(df):
    """人気指数が低い(人気がない)が総合指数がレース上位"""
    pop_rank = _rank_in_race(df, "人気指数")
    sogo_rank = _rank_in_race(df, "総合指数")
    return ((pop_rank >= 5) & (sogo_rank <= 3)).astype(int)

def f165(df):
    """厩舎印が○以上なのに人気5番以下"""
    return ((df["mark_stable_num"] >= 2) & _unpopular(df, 5)).astype(int)


# ================================================================
# カテゴリ B: オッズ・市場乖離 (7個)
# ================================================================
def f253(df):
    """当日オッズが基準より下がった(誰かが買っている) = 穴馬シグナル"""
    return (df["odds_change"] < -3).astype(int)

def f237(df):
    """基準人気順位より当日オッズ順位が悪化(市場で売られた→過小評価の可能性)"""
    day_rank = _rank_in_race(df, "day_win_odds", ascending=True)
    return ((day_rank - df["base_pop_rank"]) >= 2).astype(int)

def f246(df):
    """万券指数レース内上位3"""
    rank = _rank_in_race(df, "万券指数")
    return (rank <= 3).astype(int)

def f254(df):
    """基準複勝オッズが低い(堅実)のに単勝人気がない"""
    fuku_rank = _rank_in_race(df, "基準複勝オッズ", ascending=True)
    return ((fuku_rank <= 3) & _unpopular(df, 5)).astype(int)

def f255(df):
    """前走1番人気で凡走→今回人気急落(能力は変わっていない)"""
    return (
        (df["prev1_pop"] == 1) &
        (df["prev1_finish"] > 3) &
        _unpopular(df)
    ).astype(int)

# v2新規: 当日オッズが基準より大幅上昇 → 実は過大評価シグナル(方向反転)
def f236_rev(df):
    """当日オッズが基準比大幅上昇 = 正当な売り(過大評価だった)"""
    return (df["odds_change"] > 5).astype(int)

# v2新規: 激走指数上位 × 人気薄(条件絞り)
def f245_unpop(df):
    """激走指数レース内上位3 × 人気薄"""
    rank = _rank_in_race(df, "激走指数")
    return ((rank <= 3) & _unpopular(df)).astype(int)


# ================================================================
# カテゴリ C: スタート・不利 (8個)
# ================================================================
def f021(df):
    """前走で不利を受けた"""
    return (df["prev1_furi"].fillna(0) > 0).astype(int)

def f024(df):
    """前走で後半に不利を受けた(影響大)"""
    return (df["prev1_ato_furi"].fillna(0) > 0).astype(int)

def f026(df):
    """前走不利+着順が人気より3つ以上悪い"""
    return (
        (df["prev1_furi"].fillna(0) > 0) &
        (df["prev1_underperform"] > 3)
    ).astype(int)

def f029(df):
    """前走不利を受けたのに上がり指数上位"""
    agari_top = df["prev1_agari"].rank(ascending=False, pct=True) <= 0.25
    return ((df["prev1_furi"].fillna(0) > 0) & agari_top).astype(int)

def f025(df):
    """近走(1-2走前)で不利を2回受けた"""
    furi1 = df["prev1_furi"].fillna(0) > 0
    furi2 = df["prev2_furi"].fillna(0) > 0
    return (furi1 & furi2).astype(int)

def f033(df):
    """前走のレース評価が低いが素点は高い(展開負け)"""
    return (df["prev1_race_eval"].fillna(0) < df["prev1_soten"].fillna(0)).astype(int)

def f014(df):
    """馬スタート指数が上位25%なのに人気薄"""
    start_top = df["start_idx"].rank(ascending=False, pct=True) <= 0.25
    return (start_top & _unpopular(df, 5)).astype(int)

# v2新規: 前走不利×人気薄(条件をAND)
def f021_unpop(df):
    """前走不利あり × 人気薄"""
    return ((df["prev1_furi"].fillna(0) > 0) & _unpopular(df, 5)).astype(int)


# ================================================================
# カテゴリ D: タイム・パフォーマンス (7個)
# ================================================================
def f044(df):
    """前走5着以下だが上がり指数が優秀"""
    return (
        (df["prev1_finish"] > 5) &
        (df["prev1_agari"].rank(ascending=False, pct=True) <= 0.2)
    ).astype(int)

def f370(df):
    """前走上がり指数トップなのに着外"""
    agari_top = df["prev1_agari"].rank(ascending=False, pct=True) <= 0.15
    return ((df["prev1_finish"] > 3) & agari_top).astype(int)

def f046(df):
    """着順改善傾向(2走前→前走)"""
    return (df["finish_trend"] < -2).astype(int)

def f048(df):
    """前走前半遅いが後半速い(差し損ね型)"""
    return (
        (df["prev1_ten"].rank(pct=True) >= 0.7) &
        (df["prev1_agari"].rank(ascending=False, pct=True) <= 0.25)
    ).astype(int)

def f042(df):
    """前走テン指数上位なのに人気薄"""
    ten_top = df["prev1_ten"].rank(ascending=False, pct=True) <= 0.2
    return (ten_top & _unpopular(df)).astype(int)

def f383(df):
    """近5走平均着順が4着以内なのに人気薄"""
    return ((df["avg_finish_5"] <= 4) & _unpopular(df)).astype(int)

# v2方向反転: 近5走好走 → 既にオッズに反映済み = 過大評価
def f384_rev(df):
    """近5走で3着以内3回以上 = 好調は既に織り込み済み(過大評価リスク)"""
    return (df["top3_count_5"] >= 3).astype(int)


# ================================================================
# カテゴリ E: 芝ダ・距離・コース適性 (7個)
# ================================================================
def f057(df):
    """芝適性◎なのに人気薄"""
    turf_apt = df["turf_apt"].astype(str).str.strip()
    return ((turf_apt == "1") & _unpopular(df, 5)).astype(int)

def f058(df):
    """ダート適性◎なのに人気薄"""
    dirt_apt = df["dirt_apt"].astype(str).str.strip()
    return ((dirt_apt == "1") & _unpopular(df, 5)).astype(int)

def f059(df):
    """父馬産駒芝連対率が高いのに芝レースで人気薄"""
    is_turf = df["surface"].astype(str).str.strip() == "1"
    return (is_turf & (df["sire_turf_rate"] >= 20) & _unpopular(df, 5)).astype(int)

def f061(df):
    """父馬産駒ダート連対率が高いのにダートレースで人気薄"""
    is_dirt = df["surface"].astype(str).str.strip() == "2"
    return (is_dirt & (df["sire_dirt_rate"] >= 20) & _unpopular(df, 5)).astype(int)

def f063(df):
    """今回の馬場(芝/ダ)での通算連対率30%以上"""
    return (df["surface_pr"] >= 0.3).astype(int)

def f081(df):
    """距離適性コードが今回距離にマッチなのに人気薄"""
    dist_apt = df["distance_apt"].astype(str).str.strip()
    return (dist_apt.isin(["1", "2"]) & _unpopular(df, 5)).astype(int)

def f094(df):
    """トラック距離連対率30%以上"""
    return (df["td_pr"] >= 0.3).astype(int)


# ================================================================
# カテゴリ F: 脚質・展開 (8個) ★pace_pred修正済
# ================================================================
def f111(df):
    """追い込み馬 + Hペース予想 = 展開有利"""
    rs = df["run_style"].astype(str).str.strip()
    pace = df["pace_pred"].astype(str).str.strip()
    return (rs.isin(["3", "4"]) & (pace == "H")).astype(int)

def f112(df):
    """逃げ馬 + Sペース予想 = 展開有利"""
    rs = df["run_style"].astype(str).str.strip()
    pace = df["pace_pred"].astype(str).str.strip()
    return ((rs == "1") & (pace == "S")).astype(int)

def f116(df):
    """Hペース成績良好 + Hペース予想 + 人気薄"""
    if "fg2_h_pace_pr" not in df.columns:
        return pd.Series(0, index=df.index)
    pace = df["pace_pred"].astype(str).str.strip()
    return ((df["fg2_h_pace_pr"] >= 0.2) & (pace == "H") & _unpopular(df)).astype(int)

def f119(df):
    """道中順位予想が後方(8以上)なのに後3F順位予想が上位(3以内)"""
    return ((df["pred_mid"] >= 8) & (df["pred_3f"] <= 3)).astype(int)

def f126(df):
    """上がり指数順位がトップ3なのに人気薄"""
    return ((df["agari_rank"] <= 3) & _unpopular(df)).astype(int)

def f114(df):
    """ペースプレッシャー高(逃げ先行3頭以上)→差し追込馬に有利"""
    if "fg2_pace_pressure" not in df.columns:
        return pd.Series(0, index=df.index)
    rs = df["run_style"].astype(str).str.strip()
    return ((df["fg2_pace_pressure"] >= 3) & rs.isin(["3", "4"])).astype(int)

# v2新規: Sペース時の差し馬 → 不利(方向-1)
def f111_rev(df):
    """差し追込馬 + Sペース予想 = 展開不利(過大評価リスク)"""
    rs = df["run_style"].astype(str).str.strip()
    pace = df["pace_pred"].astype(str).str.strip()
    return (rs.isin(["3", "4"]) & (pace == "S")).astype(int)

# v2新規: Hペース時の逃げ馬 → 不利(方向-1)
def f112_rev(df):
    """逃げ馬 + Hペース予想 = 展開不利(過大評価リスク)"""
    rs = df["run_style"].astype(str).str.strip()
    pace = df["pace_pred"].astype(str).str.strip()
    return ((rs == "1") & (pace == "H")).astype(int)


# ================================================================
# カテゴリ G: 騎手・調教師 (6個) ★BB率削除
# ================================================================
def f138(df):
    """リーディング上位騎手(20位以内)が人気薄の馬に騎乗"""
    return ((df["jockey_leading"] <= 20) & _unpopular(df, 6)).astype(int)

def f143(df):
    """騎手期待単勝率10%以上なのに人気薄"""
    return ((df["jockey_exp_wr"] >= 10) & _unpopular(df, 5)).astype(int)

def f145(df):
    """騎手指数レース内上位3なのに人気薄"""
    rank = _rank_in_race(df, "騎手指数")
    return ((rank <= 3) & _unpopular(df)).astype(int)

def f156(df):
    """騎手トラック距離成績(fg3)がレース内1位"""
    if "fg3_jtd_pr" not in df.columns:
        return pd.Series(0, index=df.index)
    rank = _rank_in_race(df, "fg3_jtd_pr")
    return (rank == 1).astype(int)

def f163(df):
    """厩舎指数レース内上位3なのに人気薄"""
    rank = _rank_in_race(df, "厩舎指数")
    return ((rank <= 3) & _unpopular(df)).astype(int)

# v2新規: 騎手調教師コンボ成績良好
def f141(df):
    """騎手×調教師コンボ連対率がレース内上位"""
    if "fg3_jt_combo_pr" not in df.columns:
        return pd.Series(0, index=df.index)
    rank = _rank_in_race(df, "fg3_jt_combo_pr")
    return ((rank <= 3) & _unpopular(df)).astype(int)


# ================================================================
# カテゴリ H: 調教 (5個) ★方向反転含む
# ================================================================
def f217(df):
    """仕上指数レース内上位3"""
    rank = _rank_in_race(df, "fitness_idx")
    return (rank <= 3).astype(int)

def f219(df):
    """仕上指数変化がプラス(改善)"""
    return (df["fitness_change"].fillna(0) > 0).astype(int)

def f225(df):
    """CHK追切指数がレース内上位3なのに人気薄"""
    rank = _rank_in_race(df, "chk_workout_idx")
    return ((rank <= 3) & _unpopular(df)).astype(int)

def f221(df):
    """調教印が○以上"""
    return (df["mark_training_num"] >= 2).astype(int)

# v2方向反転: 追切指数上位は既に人気に反映 → 過大評価シグナル
def f216_rev(df):
    """追切指数レース内上位3 = 好調教は織り込み済み(過大評価リスク)"""
    rank = _rank_in_race(df, "workout_idx")
    return (rank <= 3).astype(int)


# ================================================================
# カテゴリ I: 体重・ローテ・クラス (8個) ★grade修正
# ================================================================
def f278(df):
    """上昇度が3以上"""
    return (df["improvement"].fillna(0) >= 3).astype(int)

def f281(df):
    """叩き2走目"""
    return ((df["rotation"] >= 2) & (df["rotation"] <= 4)).astype(int)

def f295(df):
    """獲得賞金がレース上位なのに人気薄"""
    rank = _rank_in_race(df, "prize_money")
    return ((rank <= 3) & _unpopular(df)).astype(int)

def f361(df):
    """前走1番人気で4着以下→今回人気急落"""
    return (
        (df["prev1_pop"] == 1) & (df["prev1_finish"] >= 4) & _unpopular(df)
    ).astype(int)

def f367(df):
    """2走前好走→前走凡走(交互パターン)"""
    return (
        (df["prev2_finish"].fillna(99) <= 3) & (df["prev1_finish"] > 5)
    ).astype(int)

def f284(df):
    """前走のグレードが今回より高い(格上実績) ★空文字列対応"""
    prev_g = pd.to_numeric(df["prev1_grade"].astype(str).str.strip(), errors="coerce")
    curr_g = pd.to_numeric(df["grade"].astype(str).str.strip(), errors="coerce")
    # グレードコード: 小さい方が格上 (1=G1, 2=G2, ... 5=条件)
    return ((prev_g < curr_g) & prev_g.notna() & curr_g.notna()).astype(int)

def f306(df):
    """季節成績の連対率が高い"""
    if "fg2_season_pr" not in df.columns:
        return pd.Series(0, index=df.index)
    return (df["fg2_season_pr"] >= 0.3).astype(int)

def f201(df):
    """馬体重大幅増(+10kg以上)"""
    if "fg4_body_weight_change" in df.columns:
        return (df["fg4_body_weight_change"].fillna(0) >= 10).astype(int)
    return (df["weight_change"].fillna(0) >= 10).astype(int)


# ================================================================
# ファクターカタログ v2
# ================================================================
FACTOR_CATALOG = {
    # --- A: JRDB指数乖離 (12個) ---
    "F336_idm1_unpop":       {"cat": "A_JRDB指数乖離", "desc": "IDMレース内1位×人気薄",       "fn": f336, "dir": +1, "extra": False},
    "F337_idm_top25_unpop":  {"cat": "A_JRDB指数乖離", "desc": "IDM上位25%×人気薄",           "fn": f337, "dir": +1, "extra": False},
    "F345_cid_top_unpop":    {"cat": "A_JRDB指数乖離", "desc": "CID上位3×人気薄",             "fn": f345, "dir": +1, "extra": False},
    "F347_agari1_unpop":     {"cat": "A_JRDB指数乖離", "desc": "上がり指数1位×人気薄",         "fn": f347, "dir": +1, "extra": False},
    "F348_ten1_unpop":       {"cat": "A_JRDB指数乖離", "desc": "テン指数1位×人気薄",           "fn": f348, "dir": +1, "extra": False},
    "F349_pace1_unpop":      {"cat": "A_JRDB指数乖離", "desc": "ペース指数1位×人気薄",         "fn": f349, "dir": +1, "extra": False},
    "F240_idm_mark_unpop":   {"cat": "A_JRDB指数乖離", "desc": "IDM印○以上×人気薄",           "fn": f240, "dir": +1, "extra": False},
    "F241_sogo_mark_unpop":  {"cat": "A_JRDB指数乖離", "desc": "総合印○以上×人気薄",           "fn": f241, "dir": +1, "extra": False},
    "F243_upset_mark":       {"cat": "A_JRDB指数乖離", "desc": "激走印あり",                   "fn": f243, "dir": +1, "extra": False},
    "F244_longshot_mark":    {"cat": "A_JRDB指数乖離", "desc": "万券印あり",                   "fn": f244, "dir": +1, "extra": False},
    "F251_low_pop_high_sogo":{"cat": "A_JRDB指数乖離", "desc": "人気指数低×総合指数高",         "fn": f251, "dir": +1, "extra": False},
    "F165_stable_mark_unpop":{"cat": "A_JRDB指数乖離", "desc": "厩舎印○以上×人気薄",           "fn": f165, "dir": +1, "extra": False},

    # --- B: オッズ・市場乖離 (7個) ---
    "F253_odds_down":        {"cat": "B_オッズ乖離", "desc": "当日オッズ急落(買われている)",      "fn": f253, "dir": +1, "extra": False},
    "F237_pop_rank_drop":    {"cat": "B_オッズ乖離", "desc": "当日人気順位悪化(売られた→穴候補)", "fn": f237, "dir": +1, "extra": False},
    "F246_manken_top3":      {"cat": "B_オッズ乖離", "desc": "万券指数レース内上位3",            "fn": f246, "dir": +1, "extra": False},
    "F254_fuku_low_tan_high":{"cat": "B_オッズ乖離", "desc": "複勝オッズ低×単勝人気薄",         "fn": f254, "dir": +1, "extra": False},
    "F255_prev_fav_flop":    {"cat": "B_オッズ乖離", "desc": "前走1番人気凡走→今回人気落ち",     "fn": f255, "dir": +1, "extra": False},
    "F236_odds_up_overval":  {"cat": "B_オッズ乖離", "desc": "当日オッズ大幅上昇=正当な売り",    "fn": f236_rev, "dir": -1, "extra": False},
    "F245_gekiso_unpop":     {"cat": "B_オッズ乖離", "desc": "激走指数上位3×人気薄",            "fn": f245_unpop, "dir": +1, "extra": False},

    # --- C: スタート・不利 (8個) ---
    "F021_prev1_furi":       {"cat": "C_スタート不利", "desc": "前走不利あり",                   "fn": f021, "dir": +1, "extra": True},
    "F024_prev1_ato_furi":   {"cat": "C_スタート不利", "desc": "前走後半不利",                   "fn": f024, "dir": +1, "extra": True},
    "F026_furi_underperf":   {"cat": "C_スタート不利", "desc": "前走不利×着順大幅悪化",          "fn": f026, "dir": +1, "extra": True},
    "F029_furi_good_agari":  {"cat": "C_スタート不利", "desc": "前走不利×上がり優秀",            "fn": f029, "dir": +1, "extra": True},
    "F025_furi_twice":       {"cat": "C_スタート不利", "desc": "近2走で不利2回",                "fn": f025, "dir": +1, "extra": True},
    "F033_race_low_soten_hi":{"cat": "C_スタート不利", "desc": "前走展開負け(素点>レース評価)",   "fn": f033, "dir": +1, "extra": True},
    "F014_start_idx_unpop":  {"cat": "C_スタート不利", "desc": "スタート指数上位×人気薄",         "fn": f014, "dir": +1, "extra": False},
    "F021_furi_unpop":       {"cat": "C_スタート不利", "desc": "前走不利あり×人気薄",            "fn": f021_unpop, "dir": +1, "extra": True},

    # --- D: タイム・パフォーマンス (7個) ---
    "F044_prev1_bad_good_ag":{"cat": "D_タイム", "desc": "前走凡走×上がり優秀",                 "fn": f044, "dir": +1, "extra": False},
    "F370_agari_top_outside":{"cat": "D_タイム", "desc": "前走上がりトップ×着外",               "fn": f370, "dir": +1, "extra": False},
    "F046_improving_trend":  {"cat": "D_タイム", "desc": "着順改善傾向(2走前→前走)",             "fn": f046, "dir": +1, "extra": False},
    "F048_slow_start_fast_f":{"cat": "D_タイム", "desc": "前走前半遅×後半速(差し損ね)",          "fn": f048, "dir": +1, "extra": False},
    "F042_ten_top_unpop":    {"cat": "D_タイム", "desc": "前走テン上位×人気薄",                 "fn": f042, "dir": +1, "extra": False},
    "F383_avg_finish_unpop": {"cat": "D_タイム", "desc": "近5走平均4着以内×人気薄",             "fn": f383, "dir": +1, "extra": False},
    "F384_good_form_overval":{"cat": "D_タイム", "desc": "近5走3着以内3回=好調織り込み済み",      "fn": f384_rev, "dir": -1, "extra": False},

    # --- E: 適性 (7個) ---
    "F057_turf_apt_unpop":   {"cat": "E_適性", "desc": "芝適性◎×人気薄",                       "fn": f057, "dir": +1, "extra": False},
    "F058_dirt_apt_unpop":   {"cat": "E_適性", "desc": "ダート適性◎×人気薄",                    "fn": f058, "dir": +1, "extra": False},
    "F059_sire_turf_unpop":  {"cat": "E_適性", "desc": "父馬芝連対率高×芝×人気薄",              "fn": f059, "dir": +1, "extra": False},
    "F061_sire_dirt_unpop":  {"cat": "E_適性", "desc": "父馬ダ連対率高×ダート×人気薄",           "fn": f061, "dir": +1, "extra": False},
    "F063_surface_pr_high":  {"cat": "E_適性", "desc": "馬場別連対率30%以上",                   "fn": f063, "dir": +1, "extra": False},
    "F081_dist_apt_unpop":   {"cat": "E_適性", "desc": "距離適性マッチ×人気薄",                 "fn": f081, "dir": +1, "extra": False},
    "F094_td_pr_high":       {"cat": "E_適性", "desc": "トラック距離連対率30%以上",              "fn": f094, "dir": +1, "extra": False},

    # --- F: 脚質・展開 (8個) ★pace_pred修正 + 逆方向追加 ---
    "F111_oikomi_hpace":     {"cat": "F_脚質展開", "desc": "追込馬×Hペース予想",                "fn": f111, "dir": +1, "extra": False},
    "F112_nige_space":       {"cat": "F_脚質展開", "desc": "逃げ馬×Sペース予想",                "fn": f112, "dir": +1, "extra": False},
    "F116_hpace_good_hpred": {"cat": "F_脚質展開", "desc": "Hペース成績良×Hペース予想×人気薄",   "fn": f116, "dir": +1, "extra": False},
    "F119_mid_back_3f_top":  {"cat": "F_脚質展開", "desc": "道中後方×後3F予想上位",              "fn": f119, "dir": +1, "extra": False},
    "F126_agari_rank_unpop": {"cat": "F_脚質展開", "desc": "上がり指数上位3×人気薄",             "fn": f126, "dir": +1, "extra": False},
    "F114_pace_press_sashi": {"cat": "F_脚質展開", "desc": "ペースプレッシャー高×差し追込",       "fn": f114, "dir": +1, "extra": False},
    "F111_sashi_space_bad":  {"cat": "F_脚質展開", "desc": "差追込×Sペース=展開不利",            "fn": f111_rev, "dir": -1, "extra": False},
    "F112_nige_hpace_bad":   {"cat": "F_脚質展開", "desc": "逃げ馬×Hペース=展開不利",            "fn": f112_rev, "dir": -1, "extra": False},

    # --- G: 騎手・調教師 (6個) ★BB率削除 ---
    "F138_top_jockey_unpop": {"cat": "G_騎手調教師", "desc": "リーディング上位騎手×人気薄",      "fn": f138, "dir": +1, "extra": False},
    "F143_jexp_wr_unpop":    {"cat": "G_騎手調教師", "desc": "騎手期待単勝率高×人気薄",          "fn": f143, "dir": +1, "extra": False},
    "F145_jidx_top_unpop":   {"cat": "G_騎手調教師", "desc": "騎手指数上位3×人気薄",             "fn": f145, "dir": +1, "extra": False},
    "F156_jtd_pr_top":       {"cat": "G_騎手調教師", "desc": "騎手TD成績レース内1位",            "fn": f156, "dir": +1, "extra": False},
    "F163_stable_idx_unpop": {"cat": "G_騎手調教師", "desc": "厩舎指数上位3×人気薄",             "fn": f163, "dir": +1, "extra": False},
    "F141_jt_combo_unpop":   {"cat": "G_騎手調教師", "desc": "騎手×調教師コンボ上位×人気薄",     "fn": f141, "dir": +1, "extra": False},

    # --- H: 調教 (5個) ★方向反転あり ---
    "F217_fitness_top3":     {"cat": "H_調教", "desc": "仕上指数レース内上位3",                  "fn": f217, "dir": +1, "extra": False},
    "F219_fitness_improved": {"cat": "H_調教", "desc": "仕上指数変化プラス",                     "fn": f219, "dir": +1, "extra": False},
    "F225_chk_top_unpop":    {"cat": "H_調教", "desc": "CHK追切上位3×人気薄",                   "fn": f225, "dir": +1, "extra": False},
    "F221_training_mark":    {"cat": "H_調教", "desc": "調教印○以上",                           "fn": f221, "dir": +1, "extra": False},
    "F216_workout_overval":  {"cat": "H_調教", "desc": "追切上位3=好調教は織り込み済み",          "fn": f216_rev, "dir": -1, "extra": False},

    # --- I: 体重・ローテ・クラス (8個) ★grade修正 ---
    "F278_improvement_high": {"cat": "I_ローテクラス", "desc": "上昇度3以上",                    "fn": f278, "dir": +1, "extra": False},
    "F281_tataki_2":         {"cat": "I_ローテクラス", "desc": "叩き2走目",                      "fn": f281, "dir": +1, "extra": False},
    "F295_prize_top_unpop":  {"cat": "I_ローテクラス", "desc": "獲得賞金上位×人気薄",             "fn": f295, "dir": +1, "extra": False},
    "F361_prev_fav1_flop":   {"cat": "I_ローテクラス", "desc": "前走1番人気4着以下→人気急落",     "fn": f361, "dir": +1, "extra": False},
    "F367_alternating":      {"cat": "I_ローテクラス", "desc": "好走凡走交互パターン",            "fn": f367, "dir": +1, "extra": False},
    "F284_prev_higher_grade":{"cat": "I_ローテクラス", "desc": "前走が今回より上のグレード",       "fn": f284, "dir": +1, "extra": True},
    "F306_season_pr_high":   {"cat": "I_ローテクラス", "desc": "季節成績連対率30%以上",            "fn": f306, "dir": +1, "extra": False},
    "F201_weight_gain":      {"cat": "I_ローテクラス", "desc": "馬体重大幅増(+10kg)",             "fn": f201, "dir": -1, "extra": False},
}

# 全ファクター名リスト
ALL_FACTOR_NAMES = sorted(FACTOR_CATALOG.keys())

# カテゴリ別グルーピング
CATEGORIES = {}
for name, meta in FACTOR_CATALOG.items():
    cat = meta["cat"]
    if cat not in CATEGORIES:
        CATEGORIES[cat] = []
    CATEGORIES[cat].append(name)
