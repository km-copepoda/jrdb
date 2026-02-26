"""各組合せN枚ずつ固定購入、必要初期費用を逆算"""
import pickle
import numpy as np
import sys
sys.path.insert(0, ".")
from upset_ef import fast_roi_detail, compute_scores


def simulate_fixed(pkl_path, strat, units=20):
    with open(pkl_path, "rb") as f:
        detail = pickle.load(f)

    race_data = detail["race_data"]
    econ = detail["econ"]
    year_scores = detail["year_scores"]
    test_years = detail["test_years"]
    rid_to_date = detail["rid_to_date"]

    sn = strat["score"]
    pct = strat["pct"]
    ff = strat["ff"]
    fo = strat["fav1_o"]
    tname = strat["ticket"]

    # 全年の選択レースを日付順に収集
    all_races = []
    for yr in test_years:
        rf = race_data[yr]
        sc = year_scores[yr]
        if sn not in sc:
            continue
        mask = np.ones(len(rf), dtype=bool)
        if ff < 99:
            mask &= (rf["n_horses"].values <= ff)
        if fo < 99:
            mask &= (rf["fav1_odds"].values <= fo)
        all_s = sc[sn]
        filt_s = all_s[mask]
        if len(filt_s) == 0:
            continue
        cutoff = np.percentile(filt_s, 100 - pct)
        sel_mask = mask & (all_s >= cutoff)
        rids = rf["race_id"].values[sel_mask].tolist()
        race_econ = econ[yr].get(tname, {})
        for rid in rids:
            if rid in race_econ:
                cost1, pay1 = race_econ[rid]
                date = rid_to_date.get(rid, yr + "0101")
                all_races.append((date, yr, rid, cost1, pay1))

    all_races.sort(key=lambda x: (x[0], x[2]))

    # 累積収支を計算して必要初期費用を逆算
    # 購入→払戻の順序: 買う時点で手元にcost分が必要
    running_profit = 0  # 過去レースの累積利益(payout-cost)
    max_needed = 0  # 必要初期費用 = max(cost_i - running_profit_before_i)
    worst_point = 0

    race_log = []
    year_summary = {}
    prev_yr = None
    yr_cost, yr_pay = 0, 0

    for i, (date, yr, rid, cost1, pay1) in enumerate(all_races):
        cost = cost1 * units
        pay = pay1 * units

        if yr != prev_yr:
            if prev_yr is not None:
                year_summary[prev_yr] = (yr_cost, yr_pay)
            yr_cost, yr_pay = 0, 0
            prev_yr = yr

        yr_cost += cost
        yr_pay += pay

        # 購入時点で必要な額: cost - 手元の累積利益
        needed = cost - running_profit
        if needed > max_needed:
            max_needed = needed
            worst_point = i + 1

        running_profit += (pay - cost)

        race_log.append({
            "no": i + 1, "date": date, "yr": yr,
            "cost": cost, "pay": pay,
            "profit": pay - cost,
            "running_profit": running_profit,
        })

    if prev_yr is not None:
        year_summary[prev_yr] = (yr_cost, yr_pay)

    required = max_needed

    label = strat.get("label", f"{sn} top{pct}%")
    print(f"\n{'='*70}")
    print(f"  {label}  [{tname}]  × {units}枚")
    cond = ""
    if ff < 99:
        cond += f" nh≤{ff}"
    if fo < 99:
        cond += f" fo≤{fo}"
    if cond:
        print(f"  条件:{cond}")
    print(f"  対象: {len(all_races)}R ({test_years[0]}-{test_years[-1]})")
    print(f"{'='*70}")

    # 年別サマリ
    print(f"\n  {'年':>6s} {'レース':>5s} {'購入額':>14s}"
          f" {'払戻額':>14s} {'損益':>14s} {'ROI':>7s}")
    print(f"  " + "-" * 65)
    total_c, total_p, total_n = 0, 0, 0
    for yr in test_years:
        if yr not in year_summary:
            continue
        yc, yp = year_summary[yr]
        yr_races = sum(1 for r in race_log if r["yr"] == yr)
        total_c += yc
        total_p += yp
        total_n += yr_races
        roi = yp / yc * 100 if yc > 0 else 0
        print(f"  {yr:>6s} {yr_races:5d} {yc:>14,}円"
              f" {yp:>14,}円 {yp-yc:>+14,}円 {roi:6.1f}%")
    print(f"  " + "-" * 65)
    total_roi = total_p / total_c * 100 if total_c > 0 else 0
    print(f"  {'合計':>6s} {total_n:5d} {total_c:>14,}円"
          f" {total_p:>14,}円 {total_p-total_c:>+14,}円"
          f" {total_roi:6.1f}%")

    # 資金推移 (初期費用 = required で開始)
    final_profit = race_log[-1]["running_profit"] if race_log else 0
    print(f"\n  ★ 必要初期費用: {required:>14,}円")
    print(f"    (最も苦しい地点: {worst_point}R目の購入時)")
    print(f"    最終所持金: {required + final_profit:>14,}円")
    print(f"    最終利益:   {final_profit:>+14,}円")

    # 所持金推移 (初期費用=requiredで開始)
    print(f"\n  --- 所持金推移 (初期費用={required:,}円) ---")
    print(f"  {'#':>3} {'日付':>10} {'購入':>10} {'払戻':>12}"
          f" {'損益':>12} {'所持金(後)':>14} {'購入前残':>14}")
    print(f"  " + "-" * 85)
    holdings = required
    for r in race_log:
        before_buy = holdings
        after_buy = holdings - r["cost"]
        holdings = holdings - r["cost"] + r["pay"]
        marker = ""
        if after_buy <= 0:
            marker = " ★ギリギリ"
        elif r["pay"] > r["cost"] * 2:
            marker = " ◎"
        print(f"  {r['no']:3d} {r['date']:>10}"
              f" {r['cost']:>10,}円 {r['pay']:>12,}円"
              f" {r['profit']:>+12,}円 {holdings:>14,}円"
              f" {before_buy:>14,}円{marker}")

    print(f"\n  最終所持金: {holdings:>14,}円")


if __name__ == "__main__":
    print("=" * 70)
    print("  固定10枚購入シミュレーション")
    print("  (払い戻し再投資あり、必要初期費用を逆算)")
    print("=" * 70)

    strategies = [
        # E定義 6年全勝
        ("output/E_detail.pkl", {
            "score": "logodds_inv", "pct": 2, "ff": 10, "fav1_o": 99,
            "ticket": "単勝ex3",
            "label": "★全勝 logodds_inv top2% nh≤10"}),
        ("output/E_detail.pkl", {
            "score": "naive", "pct": 6, "ff": 12, "fav1_o": 1.5,
            "ticket": "複勝ex3",
            "label": "★全勝 naive top6% nh≤12 fo≤1.5"}),
        ("output/E_detail.pkl", {
            "score": "logodds_inv", "pct": 9, "ff": 12, "fav1_o": 1.5,
            "ticket": "複勝ex3",
            "label": "★全勝 logodds_inv top9% nh≤12 fo≤1.5"}),
        # E定義 馬単ex3
        ("output/E_detail.pkl", {
            "score": "p_x_factor", "pct": 1, "ff": 14, "fav1_o": 99,
            "ticket": "馬単ex3",
            "label": "惜しい p_x_factor top1% nh≤14"}),
        ("output/E_detail.pkl", {
            "score": "odds_naive", "pct": 3, "ff": 12, "fav1_o": 2.0,
            "ticket": "馬単ex3",
            "label": "惜しい odds_naive top3% nh≤12 fo≤2.0"}),
        # F定義 三連単ex3
        ("output/F_detail.pkl", {
            "score": "s2_x_odds", "pct": 30, "ff": 10, "fav1_o": 1.5,
            "ticket": "三連単ex3",
            "label": "惜しい s2_x_odds top30% nh≤10 fo≤1.5"}),
    ]

    for pkl, strat in strategies:
        simulate_fixed(pkl, strat, units=10)
