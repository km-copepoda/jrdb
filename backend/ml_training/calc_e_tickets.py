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


def calc_roi_grid(race_results, ticket_type, **kwargs):
    """
    指定券種の ROI をグリッドサーチで最適化する。

    Parameters
    ----------
    race_results : DataFrame
        レース結果 + 予測スコア
    ticket_type : str
        券種名 (quinella, quinella_ex2, trifecta, trifecta_ex2, etc.)

    Returns
    -------
    dict
        最適パラメータ, ROI, 利益額
    """
    # TODO: 実装
    raise NotImplementedError("calc_roi_grid() は未実装です")


if __name__ == "__main__":
    print("calc_e_tickets.py: 券種別ROIグリッドサーチ")
    print("使い方: python calc_e_tickets.py")
    # TODO: メインの実行ロジック
