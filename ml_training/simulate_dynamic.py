"""動的賭金シミュレーション
- A: 固定10枚 (ベースライン)
- B: スコア比例 (上位ほど多く賭ける)
- C: 累積利益連動 (所持金の一定割合)
- D: 複数戦略一致で増額
"""
import pickle
import numpy as np
import sys
sys.path.insert(0, ".")


def _collect_races(detail, strat):
    """戦略に基づいてレースを収集し、スコア付きで返す"""
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
        indices = np.where(sel_mask)[0]
        rids = rf["race_id"].values[sel_mask].tolist()
        race_econ = econ[yr].get(tname, {})

        # スコアの最大値（percentile rank計算用）
        score_max = filt_s.max()
        score_min = cutoff

        for idx, rid in zip(indices, rids):
            if rid in race_econ:
                cost1, pay1 = race_econ[rid]
                date = rid_to_date.get(rid, yr + "0101")
                raw_score = all_s[idx]
                # スコアを0-1にスケール (cutoff=0, max=1)
                if score_max > score_min:
                    score_rank = (raw_score - score_min) / (score_max - score_min)
                else:
                    score_rank = 0.5
                all_races.append((date, yr, rid, cost1, pay1, score_rank))

    all_races.sort(key=lambda x: (x[0], x[2]))
    return all_races


def _simulate(all_races, units_func, label, test_years):
    """汎用シミュレーション。units_func(i, race, state) → 購入枚数"""
    state = {"holdings": 0, "running_profit": 0, "race_no": 0,
             "wins": 0, "losses": 0}

    year_summary = {}
    prev_yr = None

    results = []
    for i, (date, yr, rid, cost1, pay1, score_rank) in enumerate(all_races):
        if yr != prev_yr:
            year_summary[yr] = {"cost": 0, "pay": 0, "n": 0, "hit": 0}
            prev_yr = yr

        units = units_func(i, (date, yr, rid, cost1, pay1, score_rank), state)
        if units <= 0:
            units = 1

        cost = cost1 * units
        pay = pay1 * units
        profit = pay - cost
        state["running_profit"] += profit
        state["race_no"] = i + 1

        if pay > 0:
            state["wins"] += 1
            state["losses"] = 0  # 連敗リセット
        else:
            state["losses"] += 1

        year_summary[yr]["cost"] += cost
        year_summary[yr]["pay"] += pay
        year_summary[yr]["n"] += 1
        if pay > 0:
            year_summary[yr]["hit"] += 1

        results.append({
            "no": i + 1, "date": date, "yr": yr, "units": units,
            "cost": cost, "pay": pay, "profit": profit,
            "running_profit": state["running_profit"],
            "score_rank": score_rank,
        })

    # 必要初期費用を逆算
    running = 0
    max_needed = 0
    for r in results:
        needed = r["cost"] - running
        if needed > max_needed:
            max_needed = needed
        running += r["profit"]

    # 出力
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  対象: {len(all_races)}R ({test_years[0]}-{test_years[-1]})")
    print(f"{'='*70}")

    print(f"\n  {'年':>6s} {'R':>4s} {'的中':>4s} {'率':>5s}"
          f" {'購入額':>14s} {'払戻額':>14s} {'損益':>14s} {'ROI':>7s}")
    print(f"  " + "-" * 65)

    tc, tp, tn, th = 0, 0, 0, 0
    for yr in test_years:
        if yr not in year_summary:
            continue
        ys = year_summary[yr]
        hr = ys["hit"] / ys["n"] if ys["n"] > 0 else 0
        roi = ys["pay"] / ys["cost"] * 100 if ys["cost"] > 0 else 0
        tc += ys["cost"]
        tp += ys["pay"]
        tn += ys["n"]
        th += ys["hit"]
        print(f"  {yr:>6s} {ys['n']:4d} {ys['hit']:4d} {hr:4.0%}"
              f" {ys['cost']:>14,}円 {ys['pay']:>14,}円"
              f" {ys['pay']-ys['cost']:>+14,}円 {roi:6.1f}%")

    print(f"  " + "-" * 65)
    t_roi = tp / tc * 100 if tc > 0 else 0
    t_hr = th / tn if tn > 0 else 0
    print(f"  {'合計':>6s} {tn:4d} {th:4d} {t_hr:4.0%}"
          f" {tc:>14,}円 {tp:>14,}円"
          f" {tp-tc:>+14,}円 {t_roi:6.1f}%")

    print(f"\n  ★ 必要初期費用: {max_needed:>14,}円")
    print(f"  ★ 最終利益:     {state['running_profit']:>+14,}円")
    print(f"  ★ 最終所持金:   {max_needed + state['running_profit']:>14,}円")

    # 枚数の分布
    units_list = [r["units"] for r in results]
    print(f"  ★ 購入枚数:     min={min(units_list)}, "
          f"max={max(units_list)}, avg={np.mean(units_list):.1f}")

    return {
        "total_cost": tc, "total_pay": tp, "total_profit": tp - tc,
        "roi": t_roi, "required": max_needed,
        "final": max_needed + state["running_profit"],
    }


def run_comparison(pkl_path, strat):
    with open(pkl_path, "rb") as f:
        detail = pickle.load(f)

    test_years = detail["test_years"]
    all_races = _collect_races(detail, strat)
    label_base = strat.get("label", f"{strat['score']} top{strat['pct']}%")
    tname = strat["ticket"]

    print(f"\n{'★'*35}")
    print(f"  {label_base} [{tname}]")
    print(f"  動的賭金戦略の比較")
    print(f"{'★'*35}")

    # --- A: 固定10枚 ---
    def fixed_10(i, race, state):
        return 10

    a = _simulate(all_races, fixed_10,
                  f"【A】固定10枚", test_years)

    # --- B: スコア比例 (5〜20枚) ---
    def score_proportional(i, race, state):
        score_rank = race[5]  # 0-1
        # 最低5枚、最高20枚、スコアに比例
        return int(5 + 15 * score_rank)

    b = _simulate(all_races, score_proportional,
                  f"【B】スコア比例 (5〜20枚)", test_years)

    # --- C: 累積利益連動 (初期10万、所持金の1/20) ---
    def profit_linked(i, race, state):
        initial = 100_000
        holdings = initial + state["running_profit"]
        cost1 = race[3]
        if cost1 <= 0:
            return 1
        budget = max(holdings // 20, cost1)  # 最低1枚分
        return max(1, int(budget // cost1))

    c = _simulate(all_races, profit_linked,
                  f"【C】累積利益連動 (元手10万, 所持金の1/20)", test_years)

    # --- D: スコア比例 × 利益連動のハイブリッド ---
    def hybrid(i, race, state):
        initial = 100_000
        holdings = initial + state["running_profit"]
        cost1 = race[3]
        score_rank = race[5]
        if cost1 <= 0:
            return 1
        # ベース = 所持金の1/20
        budget = max(holdings // 20, cost1)
        base_units = max(1, int(budget // cost1))
        # スコアで0.5x〜1.5x
        multiplier = 0.5 + score_rank
        return max(1, int(base_units * multiplier))

    d = _simulate(all_races, hybrid,
                  f"【D】ハイブリッド (利益連動×スコア比例)", test_years)

    # --- 比較表 ---
    print(f"\n{'='*70}")
    print(f"  === 比較まとめ ===")
    print(f"  {'方式':>20s} {'必要初期費用':>14s} {'6年利益':>14s}"
          f" {'ROI':>7s} {'最終所持金':>14s}")
    print(f"  " + "-" * 75)
    for name, r in [("A: 固定10枚", a), ("B: スコア比例", b),
                     ("C: 利益連動", c), ("D: ハイブリッド", d)]:
        print(f"  {name:>20s} {r['required']:>14,}円"
              f" {r['total_profit']:>+14,}円 {r['roi']:6.1f}%"
              f" {r['final']:>14,}円")


if __name__ == "__main__":
    # 複勝ex3 全勝戦略 (naive top6%)
    print("\n" + "━" * 70)
    print("  【検証1】複勝ex3 全勝戦略 (naive top6% nh≤12 fo≤1.5)")
    print("━" * 70)
    run_comparison("output/E_detail.pkl", {
        "score": "naive", "pct": 6, "ff": 12, "fav1_o": 1.5,
        "ticket": "複勝ex3",
        "label": "naive top6% nh≤12 fo≤1.5",
    })

    # 複勝ex3 全勝戦略 (logodds_inv top9%)
    print("\n\n" + "━" * 70)
    print("  【検証2】複勝ex3 全勝戦略 (logodds_inv top9% nh≤12 fo≤1.5)")
    print("━" * 70)
    run_comparison("output/E_detail.pkl", {
        "score": "logodds_inv", "pct": 9, "ff": 12, "fav1_o": 1.5,
        "ticket": "複勝ex3",
        "label": "logodds_inv top9% nh≤12 fo≤1.5",
    })

    # 三連単ex3 (s2_x_odds top30%)
    print("\n\n" + "━" * 70)
    print("  【検証3】三連単ex3 (s2_x_odds top30% nh≤10 fo≤1.5)")
    print("━" * 70)
    run_comparison("output/F_detail.pkl", {
        "score": "s2_x_odds", "pct": 30, "ff": 10, "fav1_o": 1.5,
        "ticket": "三連単ex3",
        "label": "s2_x_odds top30% nh≤10 fo≤1.5",
    })
