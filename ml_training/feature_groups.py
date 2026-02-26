"""
特徴量グループ定義: 6つのFGを定義し、experiment_features.py から呼び出す。
各FGは SQL列定義と pandas 導出関数を持つ。
"""
import numpy as np
import pandas as pd


def _to_numeric(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _calc_rates(first, second, third, out):
    """成績レコード(1着/2着/3着/着外)から勝率・連対率・出走数を計算"""
    f = _to_numeric(pd.Series(first))
    s = _to_numeric(pd.Series(second))
    t = _to_numeric(pd.Series(third))
    o = _to_numeric(pd.Series(out))
    total = f + s + t + o
    wr = np.where(total > 0, f / total, np.nan)
    pr = np.where(total > 0, (f + s) / total, np.nan)
    return pd.Series(wr, dtype=float), pd.Series(pr, dtype=float), pd.Series(total, dtype=float)


# ================================================================
# FG1: 馬場状態マッチ (騎手成績, 回り成績, 良/稍/重成績)
# ================================================================
FG1_KKA_COLS = [
    ("fg1_jc_1st", "騎手成績_1着"),
    ("fg1_jc_2nd", "騎手成績_2着"),
    ("fg1_jc_3rd", "騎手成績_3着"),
    ("fg1_jc_out", "騎手成績_着外"),
    ("fg1_turn_1st", "回り成績_1着"),
    ("fg1_turn_2nd", "回り成績_2着"),
    ("fg1_turn_3rd", "回り成績_3着"),
    ("fg1_turn_out", "回り成績_着外"),
    ("fg1_good_1st", "良成績_1着"),
    ("fg1_good_2nd", "良成績_2着"),
    ("fg1_good_3rd", "良成績_3着"),
    ("fg1_good_out", "良成績_着外"),
    ("fg1_slight_1st", "稍成績_1着"),
    ("fg1_slight_2nd", "稍成績_2着"),
    ("fg1_slight_3rd", "稍成績_3着"),
    ("fg1_slight_out", "稍成績_着外"),
    ("fg1_heavy_1st", "重成績_1着"),
    ("fg1_heavy_2nd", "重成績_2着"),
    ("fg1_heavy_3rd", "重成績_3着"),
    ("fg1_heavy_out", "重成績_着外"),
]


def derive_fg1(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 騎手-馬コンボ成績
    wr, pr, total = _calc_rates(raw["fg1_jc_1st"], raw["fg1_jc_2nd"],
                                raw["fg1_jc_3rd"], raw["fg1_jc_out"])
    out["fg1_jockey_combo_wr"] = wr.values
    out["fg1_jockey_combo_pr"] = pr.values
    out["fg1_jockey_combo_total"] = total.values

    # 回り成績
    wr, pr, _ = _calc_rates(raw["fg1_turn_1st"], raw["fg1_turn_2nd"],
                            raw["fg1_turn_3rd"], raw["fg1_turn_out"])
    out["fg1_turn_wr"] = wr.values
    out["fg1_turn_pr"] = pr.values

    # 良馬場成績
    _, g_pr, _ = _calc_rates(raw["fg1_good_1st"], raw["fg1_good_2nd"],
                             raw["fg1_good_3rd"], raw["fg1_good_out"])

    # 稍重成績
    _, s_pr, _ = _calc_rates(raw["fg1_slight_1st"], raw["fg1_slight_2nd"],
                             raw["fg1_slight_3rd"], raw["fg1_slight_out"])

    # 重/不良成績
    _, h_pr, _ = _calc_rates(raw["fg1_heavy_1st"], raw["fg1_heavy_2nd"],
                             raw["fg1_heavy_3rd"], raw["fg1_heavy_out"])

    # 湿った馬場（稍重+重）合算
    wet_f = _to_numeric(raw["fg1_slight_1st"]) + _to_numeric(raw["fg1_heavy_1st"])
    wet_s = _to_numeric(raw["fg1_slight_2nd"]) + _to_numeric(raw["fg1_heavy_2nd"])
    wet_t = _to_numeric(raw["fg1_slight_3rd"]) + _to_numeric(raw["fg1_heavy_3rd"])
    wet_o = _to_numeric(raw["fg1_slight_out"]) + _to_numeric(raw["fg1_heavy_out"])
    wet_total = wet_f + wet_s + wet_t + wet_o
    out["fg1_wet_pr"] = np.where(wet_total > 0, (wet_f + wet_s) / wet_total, np.nan)

    # 当日の馬場状態にマッチした成績を使う
    turf_cond = df["turf_condition"].astype(str).str.strip() if "turf_condition" in df.columns else pd.Series("", index=df.index)
    is_heavy = turf_cond.isin(["30", "31", "32", "40", "41", "42"])
    is_slight = turf_cond.isin(["20", "21", "22"])
    out["fg1_condition_match_pr"] = np.where(
        is_heavy, out["fg1_wet_pr"],
        np.where(is_slight, s_pr.values, g_pr.values))

    # 良馬場と重馬場の差（大きい=馬場適性に偏りあり）
    out["fg1_going_gap"] = g_pr.values - out["fg1_wet_pr"].values

    return out


# ================================================================
# FG2: ペース/季節/枠 (S/M/Hペース成績, 季節成績, 枠成績)
# ================================================================
FG2_KKA_COLS = [
    ("fg2_sp_1st", "Sペース成績_1着"),
    ("fg2_sp_2nd", "Sペース成績_2着"),
    ("fg2_sp_3rd", "Sペース成績_3着"),
    ("fg2_sp_out", "Sペース成績_着外"),
    ("fg2_mp_1st", "Mペース成績_1着"),
    ("fg2_mp_2nd", "Mペース成績_2着"),
    ("fg2_mp_3rd", "Mペース成績_3着"),
    ("fg2_mp_out", "Mペース成績_着外"),
    ("fg2_hp_1st", "Hペース成績_1着"),
    ("fg2_hp_2nd", "Hペース成績_2着"),
    ("fg2_hp_3rd", "Hペース成績_3着"),
    ("fg2_hp_out", "Hペース成績_着外"),
    ("fg2_season_1st", "季節成績_1着"),
    ("fg2_season_2nd", "季節成績_2着"),
    ("fg2_season_3rd", "季節成績_3着"),
    ("fg2_season_out", "季節成績_着外"),
    ("fg2_frame_1st", "枠成績_1着"),
    ("fg2_frame_2nd", "枠成績_2着"),
    ("fg2_frame_3rd", "枠成績_3着"),
    ("fg2_frame_out", "枠成績_着外"),
]


def derive_fg2(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Sペース成績
    s_wr, s_pr, _ = _calc_rates(raw["fg2_sp_1st"], raw["fg2_sp_2nd"],
                                raw["fg2_sp_3rd"], raw["fg2_sp_out"])
    out["fg2_s_pace_wr"] = s_wr.values
    out["fg2_s_pace_pr"] = s_pr.values

    # Mペース成績
    m_wr, m_pr, _ = _calc_rates(raw["fg2_mp_1st"], raw["fg2_mp_2nd"],
                                raw["fg2_mp_3rd"], raw["fg2_mp_out"])
    out["fg2_m_pace_wr"] = m_wr.values
    out["fg2_m_pace_pr"] = m_pr.values

    # Hペース成績
    h_wr, h_pr, _ = _calc_rates(raw["fg2_hp_1st"], raw["fg2_hp_2nd"],
                                raw["fg2_hp_3rd"], raw["fg2_hp_out"])
    out["fg2_h_pace_wr"] = h_wr.values
    out["fg2_h_pace_pr"] = h_pr.values

    # ペース予想にマッチした成績
    pace = df["pace_pred"].astype(str).str.strip() if "pace_pred" in df.columns else pd.Series("", index=df.index)
    out["fg2_pace_match_pr"] = np.where(
        pace == "1", s_pr.values,
        np.where(pace == "3", h_pr.values, m_pr.values))

    # ペース汎用性（各ペースの連対率のばらつきが小さい=万能型）
    pace_stack = np.column_stack([s_pr.values, m_pr.values, h_pr.values])
    out["fg2_pace_versatility"] = np.nanstd(pace_stack, axis=1)

    # 季節成績
    _, sea_pr, _ = _calc_rates(raw["fg2_season_1st"], raw["fg2_season_2nd"],
                               raw["fg2_season_3rd"], raw["fg2_season_out"])
    out["fg2_season_pr"] = sea_pr.values

    # 枠成績
    _, frm_pr, frm_total = _calc_rates(raw["fg2_frame_1st"], raw["fg2_frame_2nd"],
                                       raw["fg2_frame_3rd"], raw["fg2_frame_out"])
    out["fg2_frame_pr"] = frm_pr.values
    out["fg2_frame_total"] = frm_total.values

    # レース内の脚質分布（逃げ+先行の数 = ペースプレッシャー）
    if "run_style" in df.columns:
        rs = df["run_style"].astype(str).str.strip()
        is_front = rs.isin(["1", "2"])  # 1=逃げ, 2=先行
        pace_pressure = df.groupby("race_id")["run_style"].transform(
            lambda x: x.astype(str).str.strip().isin(["1", "2"]).sum()
        )
        out["fg2_pace_pressure"] = _to_numeric(pace_pressure)

    return out


# ================================================================
# FG3: 騎手専門性 (騎手距離成績, 騎手TD成績, 騎手調教師別成績)
# ================================================================
FG3_KKA_COLS = [
    ("fg3_jd_1st", "騎手距離成績_1着"),
    ("fg3_jd_2nd", "騎手距離成績_2着"),
    ("fg3_jd_3rd", "騎手距離成績_3着"),
    ("fg3_jd_out", "騎手距離成績_着外"),
    ("fg3_jtd_1st", "騎手トラック距離成績_1着"),
    ("fg3_jtd_2nd", "騎手トラック距離成績_2着"),
    ("fg3_jtd_3rd", "騎手トラック距離成績_3着"),
    ("fg3_jtd_out", "騎手トラック距離成績_着外"),
    ("fg3_jt_1st", "騎手調教師別成績_1着"),
    ("fg3_jt_2nd", "騎手調教師別成績_2着"),
    ("fg3_jt_3rd", "騎手調教師別成績_3着"),
    ("fg3_jt_out", "騎手調教師別成績_着外"),
]


def derive_fg3(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 騎手-距離成績
    wr, pr, total = _calc_rates(raw["fg3_jd_1st"], raw["fg3_jd_2nd"],
                                raw["fg3_jd_3rd"], raw["fg3_jd_out"])
    out["fg3_jdist_wr"] = wr.values
    out["fg3_jdist_pr"] = pr.values
    out["fg3_jdist_total"] = total.values

    # 騎手-トラック距離成績（最も限定的）
    wr, pr, total = _calc_rates(raw["fg3_jtd_1st"], raw["fg3_jtd_2nd"],
                                raw["fg3_jtd_3rd"], raw["fg3_jtd_out"])
    out["fg3_jtd_wr"] = wr.values
    out["fg3_jtd_pr"] = pr.values
    out["fg3_jtd_total"] = total.values

    # 騎手-調教師コンボ成績
    wr, pr, total = _calc_rates(raw["fg3_jt_1st"], raw["fg3_jt_2nd"],
                                raw["fg3_jt_3rd"], raw["fg3_jt_out"])
    out["fg3_jt_combo_wr"] = wr.values
    out["fg3_jt_combo_pr"] = pr.values

    # 騎手の全体勝率との乖離（大きい=この条件で不慣れ）
    if "jockey_wr" in df.columns:
        out["fg3_jockey_general_vs_specific"] = df["jockey_wr"].values - out["fg3_jtd_wr"].values

    return out


# ================================================================
# FG4: 馬体重 (TYB: 馬体重, 馬体重増減)
# ================================================================
FG4_TYB_COLS = [
    ("fg4_body_weight", "馬体重"),
    ("fg4_body_weight_change", "馬体重増減"),
]


def derive_fg4(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    bw = _to_numeric(raw["fg4_body_weight"])
    bwc = _to_numeric(raw["fg4_body_weight_change"])

    out["fg4_body_weight"] = bw
    out["fg4_body_weight_change"] = bwc
    out["fg4_body_weight_abs_change"] = bwc.abs()
    out["fg4_weight_change_severe"] = (bwc.abs() >= 12).astype(float)

    # レース内での馬体重の相対値
    if "race_id" in df.columns:
        out["fg4_weight_vs_race_mean"] = bw - bw.groupby(df["race_id"]).transform("mean")

    return out


# ================================================================
# FG5: 重量種別 + ローテ成績 (BAC: 重量, KKA: ローテ成績)
# ================================================================
FG5_BAC_COLS = [
    ("fg5_weight_type", "重量"),
]

FG5_KKA_COLS = [
    ("fg5_rote_1st", "ローテ成績_1着"),
    ("fg5_rote_2nd", "ローテ成績_2着"),
    ("fg5_rote_3rd", "ローテ成績_3着"),
    ("fg5_rote_out", "ローテ成績_着外"),
]


def derive_fg5(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 重量種別
    wt = raw["fg5_weight_type"].astype(str).str.strip() if "fg5_weight_type" in raw.columns else pd.Series("", index=df.index)
    out["fg5_is_handicap"] = (wt == "1").astype(float)
    out["fg5_weight_type_num"] = _to_numeric(wt)

    # ハンデ × レースの混戦度
    if "race_day_entropy" in df.columns:
        out["fg5_handicap_x_entropy"] = out["fg5_is_handicap"] * df["race_day_entropy"].values

    # ローテ成績
    wr, pr, total = _calc_rates(raw["fg5_rote_1st"], raw["fg5_rote_2nd"],
                                raw["fg5_rote_3rd"], raw["fg5_rote_out"])
    out["fg5_rote_wr"] = wr.values
    out["fg5_rote_pr"] = pr.values
    out["fg5_rote_total"] = total.values

    return out


# ================================================================
# FG6: コンボ成績 + レース集約 (騎手馬主別, 騎手ブリンカ, 調教師馬主別)
# ================================================================
FG6_KKA_COLS = [
    ("fg6_jo_1st", "騎手馬主別成績_1着"),
    ("fg6_jo_2nd", "騎手馬主別成績_2着"),
    ("fg6_jo_3rd", "騎手馬主別成績_3着"),
    ("fg6_jo_out", "騎手馬主別成績_着外"),
    ("fg6_jb_1st", "騎手ブリンカ成績_1着"),
    ("fg6_jb_2nd", "騎手ブリンカ成績_2着"),
    ("fg6_jb_3rd", "騎手ブリンカ成績_3着"),
    ("fg6_jb_out", "騎手ブリンカ成績_着外"),
    ("fg6_to_1st", "調教師馬主別成績_1着"),
    ("fg6_to_2nd", "調教師馬主別成績_2着"),
    ("fg6_to_3rd", "調教師馬主別成績_3着"),
    ("fg6_to_out", "調教師馬主別成績_着外"),
]


def derive_fg6(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 騎手-馬主コンボ
    wr, pr, _ = _calc_rates(raw["fg6_jo_1st"], raw["fg6_jo_2nd"],
                            raw["fg6_jo_3rd"], raw["fg6_jo_out"])
    out["fg6_jowner_wr"] = wr.values
    out["fg6_jowner_pr"] = pr.values

    # 騎手ブリンカ成績
    wr, pr, _ = _calc_rates(raw["fg6_jb_1st"], raw["fg6_jb_2nd"],
                            raw["fg6_jb_3rd"], raw["fg6_jb_out"])
    out["fg6_jblinker_wr"] = wr.values
    out["fg6_jblinker_pr"] = pr.values

    # 調教師-馬主コンボ
    wr, pr, _ = _calc_rates(raw["fg6_to_1st"], raw["fg6_to_2nd"],
                            raw["fg6_to_3rd"], raw["fg6_to_out"])
    out["fg6_towner_wr"] = wr.values
    out["fg6_towner_pr"] = pr.values

    # レース内集約: surface_pr のレース内偏差値
    if "surface_pr" in df.columns:
        race_mean = df.groupby("race_id")["surface_pr"].transform("mean")
        race_std = df.groupby("race_id")["surface_pr"].transform("std")
        out["fg6_surface_pr_zscore"] = np.where(
            race_std > 0, (df["surface_pr"] - race_mean) / race_std, 0)

    # td_pr のレース内偏差値
    if "td_pr" in df.columns:
        race_mean = df.groupby("race_id")["td_pr"].transform("mean")
        race_std = df.groupby("race_id")["td_pr"].transform("std")
        out["fg6_td_pr_zscore"] = np.where(
            race_std > 0, (df["td_pr"] - race_mean) / race_std, 0)

    # 総合印のレース内ランク
    if "mark_overall_num" in df.columns:
        out["fg6_mark_rank_in_race"] = df.groupby("race_id")["mark_overall_num"].rank(
            ascending=False, method="min")

    return out


# ================================================================
# レジストリ
# ================================================================
FEATURE_GROUPS = {
    "FG1": {
        "name": "馬場状態マッチ (騎手成績, 回り, 良/稍/重)",
        "kka_cols": FG1_KKA_COLS,
        "tyb_cols": [],
        "bac_cols": [],
        "derive_fn": derive_fg1,
    },
    "FG2": {
        "name": "ペース/季節/枠",
        "kka_cols": FG2_KKA_COLS,
        "tyb_cols": [],
        "bac_cols": [],
        "derive_fn": derive_fg2,
    },
    "FG3": {
        "name": "騎手専門性 (距離, TD, 調教師)",
        "kka_cols": FG3_KKA_COLS,
        "tyb_cols": [],
        "bac_cols": [],
        "derive_fn": derive_fg3,
    },
    "FG4": {
        "name": "馬体重",
        "kka_cols": [],
        "tyb_cols": FG4_TYB_COLS,
        "bac_cols": [],
        "derive_fn": derive_fg4,
    },
    "FG5": {
        "name": "重量種別 + ローテ",
        "kka_cols": FG5_KKA_COLS,
        "tyb_cols": [],
        "bac_cols": FG5_BAC_COLS,
        "derive_fn": derive_fg5,
    },
    "FG6": {
        "name": "コンボ + レース集約",
        "kka_cols": FG6_KKA_COLS,
        "tyb_cols": [],
        "bac_cols": [],
        "derive_fn": derive_fg6,
    },
}

ALL_FG_NAMES = sorted(FEATURE_GROUPS.keys())
