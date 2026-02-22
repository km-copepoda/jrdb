"""
券種別ROIグリッドサーチ

対象券種 (12種):
- 馬連 (全通り)
- 馬連 (ex2): 1, 2番人気除外
- 馬連 (ex2_1av): 1, 2番人気が1頭だけ入る
- 単勝 (全通り)
- 三連複 (全通り)
- 三連複 (ex1): 1番人気除外
- 三連複 (ex2): 1, 2番人気除外
- 三連複 (ex3): 1, 2, 3番人気除外

ベストパフォーマンス (2024-2026ローリング実績):
- 馬連 ex2 | rest_threat top3% | r_nh16 h14 f4.0: 167.4% (+75.2万円)
- 三連複 ex2 | 利益最大化 | rest_threat top3%: 161.7% (+56.1万円)
"""

import itertools
import numpy as np
import pandas as pd
import psycopg2
from config import DB_CONFIG, TABLES


def _get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_payouts():
    """HUC (払戻情報) をDBから読み込む"""
    conn = _get_connection()
    sql = f"""
    SELECT
        hjc."前日_番組情報_id" AS race_id,
        hjc."単勝払戻1_馬番"   AS win1_umaban,
        hjc."単勝払戻1_払戻金" AS win1_payout,
        hjc."馬連払戻1_馬番組合せ" AS quinella1_combo,
        hjc."馬連払戻1_払戻金"    AS quinella1_payout,
        hjc."馬連払戻2_馬番組合せ" AS quinella2_combo,
        hjc."馬連払戻2_払戻金"    AS quinella2_payout,
        hjc."馬連払戻3_馬番組合せ" AS quinella3_combo,
        hjc."馬連払戻3_払戻金"    AS quinella3_payout,
        hjc."三連複払戻1_馬番組合せ" AS trio1_combo,
        hjc."三連複払戻1_払戻金"    AS trio1_payout,
        hjc."三連複払戻2_馬番組合せ" AS trio2_combo,
        hjc."三連複払戻2_払戻金"    AS trio2_payout,
        hjc."三連複払戻3_馬番組合せ" AS trio3_combo,
        hjc."三連複払戻3_払戻金"    AS trio3_payout
    FROM {TABLES['HUC']} AS hjc
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def _parse_combo(combo_str, n=2):
    """馬番組合せ文字列をパース (例: '0305' -> (3, 5))"""
    if not combo_str or not combo_str.strip():
        return None
    s = combo_str.strip()
    if n == 2 and len(s) >= 4:
        a, b = int(s[:2]), int(s[2:4])
        return tuple(sorted([a, b]))
    elif n == 3 and len(s) >= 6:
        a, b, c = int(s[:2]), int(s[2:4]), int(s[4:6])
        return tuple(sorted([a, b, c]))
    return None


def _safe_int(val):
    """安全に整数変換"""
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return 0


def build_ticket_data(horse_df, race_scored_df, payouts_df):
    """
    馬レベル予測 + レーススコア + 払戻データを統合して券種計算用データを構築

    Parameters
    ----------
    horse_df : DataFrame
        馬レベルの予測 (pred, umaban, race_id, tyb_win_odds, finish_pos, ...)
    race_scored_df : DataFrame
        レーススコア (race_id, rest_threat, blend50, fav1_umaban, ...)
    payouts_df : DataFrame
        HUC払戻情報

    Returns
    -------
    merged : DataFrame
    """
    merged = race_scored_df.merge(payouts_df, on="race_id", how="inner")
    return merged, horse_df


def _generate_quinella_combos(horse_df, race_id, exclude_top_n=0, fav_one_in=False):
    """
    馬連の組合せを生成

    exclude_top_n: 上位N頭を除外
    fav_one_in: True の場合、除外対象から1頭だけ入れる
    """
    grp = horse_df[horse_df["race_id"] == race_id].sort_values("tyb_win_odds")
    if len(grp) < 2:
        return []

    all_umaban = grp["umaban"].astype(int).tolist()
    top_umaban = set(all_umaban[:exclude_top_n]) if exclude_top_n > 0 else set()
    rest_umaban = [u for u in all_umaban if u not in top_umaban]

    if fav_one_in and top_umaban:
        combos = []
        for fav in top_umaban:
            for r in rest_umaban:
                combos.append(tuple(sorted([fav, r])))
        return list(set(combos))

    if exclude_top_n > 0:
        return list(itertools.combinations(sorted(rest_umaban), 2))

    return list(itertools.combinations(sorted(all_umaban), 2))


def _generate_trio_combos(horse_df, race_id, exclude_top_n=0):
    """三連複の組合せを生成"""
    grp = horse_df[horse_df["race_id"] == race_id].sort_values("tyb_win_odds")
    if len(grp) < 3:
        return []

    all_umaban = grp["umaban"].astype(int).tolist()
    if exclude_top_n > 0:
        rest = all_umaban[exclude_top_n:]
    else:
        rest = all_umaban

    return list(itertools.combinations(sorted(rest), 3))


def calc_roi_grid(horse_df, race_scored_df, payouts_df, ticket_type, **kwargs):
    """
    指定券種の ROI をグリッドサーチで最適化する。

    Parameters
    ----------
    horse_df : DataFrame
        馬レベルの予測データ
    race_scored_df : DataFrame
        レーススコア (rest_threat, blend50 等)
    payouts_df : DataFrame
        払戻情報
    ticket_type : str
        券種名

    Returns
    -------
    dict : 最適パラメータ, ROI, 利益額
    """
    merged = race_scored_df.merge(payouts_df, on="race_id", how="inner")

    score_cols = ["rest_threat", "blend50", "naive"]
    top_pcts = [0.01, 0.02, 0.03, 0.05, 0.10]

    best_result = {"roi": 0, "profit": 0}

    for score_col in score_cols:
        for top_pct in top_pcts:
            threshold = merged[score_col].quantile(1 - top_pct)
            target_races = merged[merged[score_col] >= threshold]

            total_cost = 0
            total_payout = 0

            for _, race in target_races.iterrows():
                race_id = race["race_id"]

                if ticket_type == "win":
                    # 単勝: 全通り
                    grp = horse_df[horse_df["race_id"] == race_id]
                    n_bets = len(grp)
                    total_cost += n_bets * 100
                    win_umaban = _safe_int(race.get("win1_umaban", 0))
                    if win_umaban > 0:
                        total_payout += _safe_int(race.get("win1_payout", 0))

                elif ticket_type.startswith("quinella"):
                    # 馬連
                    exclude = 0
                    fav_one = False
                    if "ex2" in ticket_type and "1av" in ticket_type:
                        exclude = 2
                        fav_one = True
                    elif "ex2" in ticket_type:
                        exclude = 2

                    combos = _generate_quinella_combos(
                        horse_df, race_id, exclude, fav_one
                    )
                    total_cost += len(combos) * 100

                    # 的中判定
                    for i in range(1, 4):
                        combo_col = f"quinella{i}_combo"
                        payout_col = f"quinella{i}_payout"
                        hit_combo = _parse_combo(race.get(combo_col, ""), n=2)
                        if hit_combo and hit_combo in combos:
                            total_payout += _safe_int(race.get(payout_col, 0))

                elif ticket_type.startswith("trio"):
                    # 三連複
                    exclude = 0
                    if "ex3" in ticket_type:
                        exclude = 3
                    elif "ex2" in ticket_type:
                        exclude = 2
                    elif "ex1" in ticket_type:
                        exclude = 1

                    combos = _generate_trio_combos(horse_df, race_id, exclude)
                    total_cost += len(combos) * 300

                    for i in range(1, 4):
                        combo_col = f"trio{i}_combo"
                        payout_col = f"trio{i}_payout"
                        hit_combo = _parse_combo(race.get(combo_col, ""), n=3)
                        if hit_combo and hit_combo in [tuple(sorted(c)) for c in combos]:
                            total_payout += _safe_int(race.get(payout_col, 0))

            if total_cost > 0:
                roi = total_payout / total_cost * 100
                profit = total_payout - total_cost

                if roi > best_result["roi"]:
                    best_result = {
                        "ticket_type": ticket_type,
                        "score_col": score_col,
                        "top_pct": top_pct,
                        "roi": roi,
                        "profit": profit,
                        "total_cost": total_cost,
                        "total_payout": total_payout,
                        "n_races": len(target_races),
                    }

    return best_result


def main():
    print("=== calc_e_tickets.py: 券種別ROIグリッドサーチ ===\n")

    # 予測結果を読み込み
    try:
        horse_df = pd.read_csv("predictions.csv")
    except FileNotFoundError:
        print("predictions.csv が見つかりません。先に train.py を実行してください。")
        return

    # upset_scores を読み込み
    try:
        race_scored_df = pd.read_csv("upset_scores.csv")
    except FileNotFoundError:
        print("upset_scores.csv が見つかりません。先に upset_ef.py を実行してください。")
        return

    # 払戻情報を読み込み
    print("払戻情報を読み込み中...")
    payouts_df = load_payouts()
    print(f"  {len(payouts_df)} レースの払戻情報\n")

    # 券種一覧
    ticket_types = [
        "win",               # 単勝 (全通り)
        "quinella",          # 馬連 (全通り)
        "quinella_ex2",      # 馬連 (1,2番人気除外)
        "quinella_ex2_1av",  # 馬連 (1,2番人気1頭だけ)
        "trio",              # 三連複 (全通り)
        "trio_ex1",          # 三連複 (1番人気除外)
        "trio_ex2",          # 三連複 (1,2番人気除外)
        "trio_ex3",          # 三連複 (1,2,3番人気除外)
    ]

    results = []
    for tt in ticket_types:
        print(f"計算中: {tt}...")
        result = calc_roi_grid(horse_df, race_scored_df, payouts_df, tt)
        results.append(result)
        if result["roi"] > 0:
            print(f"  -> ROI: {result['roi']:.1f}%, "
                  f"利益: {result['profit']/10000:.1f}万円, "
                  f"スコア: {result.get('score_col', 'N/A')} top{result.get('top_pct', 0):.0%}")
        else:
            print(f"  -> 利益なし")

    # 結果サマリー
    print("\n" + "=" * 70)
    print("券種別ベスト結果")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roi", ascending=False)

    for _, row in results_df.iterrows():
        if row["roi"] > 0:
            print(
                f"  {row.get('ticket_type', 'N/A'):20s} | "
                f"ROI: {row['roi']:6.1f}% | "
                f"利益: {row['profit']/10000:+8.1f}万円 | "
                f"{row.get('score_col', 'N/A')} top{row.get('top_pct', 0):.0%} | "
                f"{row.get('n_races', 0)}レース"
            )

    results_df.to_csv("ticket_results.csv", index=False)
    print("\n結果を ticket_results.csv に保存しました。")


if __name__ == "__main__":
    main()
