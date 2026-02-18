from django.db import models

from database.models.PreviousDayInformation import (
    前日_番組情報,
    前日_競走馬情報,
    前日_開催情報,
    馬基本情報,
)
from .helpers import (
    場コード,
    馬場状態コード,
    種別コード,
    重量コード,
    グレードコード,
    異常区分コード,
    コース取りコード,
    上昇度コード,
    クラスコード,
    馬体コード,
    気配コード,
    天候コード,
    脚質コード,
)


class 成績_払戻情報(models.Model):
    """
    file name is HJC
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="成績_払戻情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="成績_払戻情報-前日_番組情報_IDX"),
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    前日_番組情報 = models.OneToOneField(
        前日_番組情報, on_delete=models.CASCADE, max_length=8, primary_key=True, db_index=False
    )
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    単勝払戻1_馬番 = models.CharField(max_length=2)
    単勝払戻1_払戻金 = models.CharField(max_length=7)
    単勝払戻2_馬番 = models.CharField(max_length=2)
    単勝払戻2_払戻金 = models.CharField(max_length=7)
    単勝払戻3_馬番 = models.CharField(max_length=2)
    単勝払戻3_払戻金 = models.CharField(max_length=7)
    複勝払戻1_馬番 = models.CharField(max_length=2)
    複勝払戻1_払戻金 = models.CharField(max_length=7)
    複勝払戻2_馬番 = models.CharField(max_length=2)
    複勝払戻2_払戻金 = models.CharField(max_length=7)
    複勝払戻3_馬番 = models.CharField(max_length=2)
    複勝払戻3_払戻金 = models.CharField(max_length=7)
    複勝払戻4_馬番 = models.CharField(max_length=2)
    複勝払戻4_払戻金 = models.CharField(max_length=7)
    複勝払戻5_馬番 = models.CharField(max_length=2)
    複勝払戻5_払戻金 = models.CharField(max_length=7)
    枠連払戻1_枠番組合せ = models.CharField(max_length=2)
    枠連払戻1_払戻金 = models.CharField(max_length=7)
    枠連払戻2_枠番組合せ = models.CharField(max_length=2)
    枠連払戻2_払戻金 = models.CharField(max_length=7)
    枠連払戻3_枠番組合せ = models.CharField(max_length=2)
    枠連払戻3_払戻金 = models.CharField(max_length=7)
    馬連払戻1_馬番組合せ = models.CharField(max_length=4)
    馬連払戻1_払戻金 = models.CharField(max_length=8)
    馬連払戻2_馬番組合せ = models.CharField(max_length=4)
    馬連払戻2_払戻金 = models.CharField(max_length=8)
    馬連払戻3_馬番組合せ = models.CharField(max_length=4)
    馬連払戻3_払戻金 = models.CharField(max_length=8)
    ワイド払戻1_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻1_払戻金 = models.CharField(max_length=8)
    ワイド払戻2_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻2_払戻金 = models.CharField(max_length=8)
    ワイド払戻3_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻3_払戻金 = models.CharField(max_length=8)
    ワイド払戻4_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻4_払戻金 = models.CharField(max_length=8)
    ワイド払戻5_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻5_払戻金 = models.CharField(max_length=8)
    ワイド払戻6_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻6_払戻金 = models.CharField(max_length=8)
    ワイド払戻7_馬番組合せ = models.CharField(max_length=4)
    ワイド払戻7_払戻金 = models.CharField(max_length=8)
    馬単払戻1_馬番組合せ = models.CharField(max_length=4)
    馬単払戻1_払戻金 = models.CharField(max_length=8)
    馬単払戻2_馬番組合せ = models.CharField(max_length=4)
    馬単払戻2_払戻金 = models.CharField(max_length=8)
    馬単払戻3_馬番組合せ = models.CharField(max_length=4)
    馬単払戻3_払戻金 = models.CharField(max_length=8)
    馬単払戻4_馬番組合せ = models.CharField(max_length=4)
    馬単払戻4_払戻金 = models.CharField(max_length=8)
    馬単払戻5_馬番組合せ = models.CharField(max_length=4)
    馬単払戻5_払戻金 = models.CharField(max_length=8)
    馬単払戻6_馬番組合せ = models.CharField(max_length=4)
    馬単払戻6_払戻金 = models.CharField(max_length=8)
    三連複払戻1_馬番組合せ = models.CharField(max_length=6)
    三連複払戻1_払戻金 = models.CharField(max_length=8)
    三連複払戻2_馬番組合せ = models.CharField(max_length=6)
    三連複払戻2_払戻金 = models.CharField(max_length=8)
    三連複払戻3_馬番組合せ = models.CharField(max_length=6)
    三連複払戻3_払戻金 = models.CharField(max_length=8)
    三連単払戻1_馬番組合せ = models.CharField(max_length=6)
    三連単払戻1_払戻金 = models.CharField(max_length=9)
    三連単払戻2_馬番組合せ = models.CharField(max_length=6)
    三連単払戻2_払戻金 = models.CharField(max_length=9)
    三連単払戻3_馬番組合せ = models.CharField(max_length=6)
    三連単払戻3_払戻金 = models.CharField(max_length=9)
    三連単払戻4_馬番組合せ = models.CharField(max_length=6)
    三連単払戻4_払戻金 = models.CharField(max_length=9)
    三連単払戻5_馬番組合せ = models.CharField(max_length=6)
    三連単払戻5_払戻金 = models.CharField(max_length=9)
    三連単払戻6_馬番組合せ = models.CharField(max_length=6)
    三連単払戻6_払戻金 = models.CharField(max_length=9)
    予備 = models.CharField(max_length=11)


class 成績_成績レース情報(models.Model):
    """
    file name is SRB
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="成績_成績レース情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="成績_成績レース情報-前日_番組情報_IDX"),
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    前日_番組情報 = models.OneToOneField(
        前日_番組情報, on_delete=models.CASCADE, max_length=8, primary_key=True, db_index=False
    )
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    ハロンタイム = models.CharField(max_length=54)
    一コーナー = models.CharField(max_length=64)
    二コーナー = models.CharField(max_length=64)
    三コーナー = models.CharField(max_length=64)
    四コーナー = models.CharField(max_length=64)
    ペースアップ位置 = models.CharField(max_length=2)
    一角 = models.CharField(max_length=3)
    二角 = models.CharField(max_length=3)
    向正 = models.CharField(max_length=3)
    三角 = models.CharField(max_length=3)
    四角 = models.CharField(max_length=5)
    直線 = models.CharField(max_length=5)
    レースコメント = models.TextField(max_length=500)
    予備 = models.CharField(max_length=8)


class 成績_成績分析用拡張情報(models.Model):
    """
    file name is SKB
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="成績_成績分析用拡張情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="成績_成績分析用拡張情報-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="成績_成績分析用拡張情報-前日_競走馬情報_IDX"),
            models.Index(fields=["馬基本情報"], name="成績_成績分析用拡張情報-馬基本情報_IDX"),
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    前日_番組情報 = models.ForeignKey(前日_番組情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    前日_競走馬情報 = models.OneToOneField(
        前日_競走馬情報, on_delete=models.CASCADE, max_length=10, primary_key=True, db_index=False
    )
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    馬番 = models.CharField(max_length=2)
    馬基本情報 = models.ForeignKey(馬基本情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    年月日 = models.CharField(max_length=8)
    特記コード = models.CharField(max_length=18)
    馬具コード = models.CharField(max_length=24)
    総合 = models.CharField(max_length=9)
    左前 = models.CharField(max_length=9)
    右前 = models.CharField(max_length=9)
    左後 = models.CharField(max_length=9)
    右後 = models.CharField(max_length=9)
    パドックコメント = models.CharField(max_length=40)
    脚元コメント = models.CharField(max_length=40)
    馬具その他コメント = models.CharField(max_length=40)
    レースコメント = models.CharField(max_length=40)
    ハミ = models.CharField(max_length=3)
    バンテージ = models.CharField(max_length=3)
    蹄鉄 = models.CharField(max_length=3)
    蹄状態 = models.CharField(max_length=3)
    ソエ = models.CharField(max_length=3)
    骨瘤 = models.CharField(max_length=3)
    予備 = models.CharField(max_length=11)


class 成績_成績分析用情報(models.Model):
    """
    file name is SED
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="成績_成績分析用情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="成績_成績分析用情報-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="成績_成績分析用情報-前日_競走馬情報_IDX"),
            models.Index(fields=["馬基本情報"], name="成績_成績分析用情報-馬基本情報_IDX"),
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    前日_番組情報 = models.ForeignKey(前日_番組情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    前日_競走馬情報 = models.OneToOneField(
        前日_競走馬情報, on_delete=models.CASCADE, max_length=10, primary_key=True, db_index=False
    )
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    馬番 = models.CharField(max_length=2)
    馬基本情報 = models.ForeignKey(馬基本情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    年月日 = models.CharField(max_length=8)
    馬名 = models.CharField(max_length=36)
    距離 = models.CharField(max_length=4)
    芝ダ障害コード = models.CharField(max_length=1)
    右左 = models.CharField(max_length=1)
    内外 = models.CharField(max_length=1)
    馬場状態コード = models.CharField(choices=馬場状態コード, max_length=2)
    種別コード = models.CharField(choices=種別コード, max_length=2)
    条件 = models.CharField(max_length=2)
    記号 = models.CharField(max_length=3)
    重量コード = models.CharField(choices=重量コード, max_length=1)
    グレードコード = models.CharField(choices=グレードコード, max_length=1)
    レース名 = models.CharField(max_length=50)
    頭数 = models.CharField(max_length=2)
    レース名略称 = models.CharField(max_length=8)
    着順 = models.CharField(max_length=2)
    異常区分コード = models.CharField(choices=異常区分コード, max_length=1)
    タイム = models.CharField(max_length=4)
    斤量 = models.CharField(max_length=3)
    騎手名 = models.CharField(max_length=12)
    調教師名 = models.CharField(max_length=12)
    確定単勝オッズ = models.CharField(max_length=6)
    確定単勝人気順位 = models.CharField(max_length=2)
    IDM = models.CharField(max_length=3)
    素点 = models.CharField(max_length=3)
    馬場差 = models.CharField(max_length=3)
    ペース = models.CharField(max_length=3)
    出遅 = models.CharField(max_length=3)
    位置取 = models.CharField(max_length=3)
    不利 = models.CharField(max_length=3)
    前不利 = models.CharField(max_length=3)
    中不利 = models.CharField(max_length=3)
    後不利 = models.CharField(max_length=3)
    レース = models.CharField(max_length=3)
    コース取りコード = models.CharField(choices=コース取りコード, max_length=1)
    上昇度コード = models.CharField(choices=上昇度コード, max_length=1)
    クラスコード = models.CharField(choices=クラスコード, max_length=2)
    馬体コード = models.CharField(choices=馬体コード, max_length=1)
    気配コード = models.CharField(choices=気配コード, max_length=1)
    レースペース = models.CharField(max_length=1)
    馬ペース = models.CharField(max_length=1)
    テン指数 = models.CharField(max_length=5)
    上がり指数 = models.CharField(max_length=5)
    ペース指数 = models.CharField(max_length=5)
    レースP指数 = models.CharField(max_length=5)
    一二着馬名 = models.CharField(max_length=12)
    一二着タイム差 = models.CharField(max_length=3)
    前3Fタイム = models.CharField(max_length=3)
    後3Fタイム = models.CharField(max_length=3)
    備考 = models.CharField(max_length=24)
    予備1 = models.CharField(max_length=2)
    確定複勝オッズ下 = models.CharField(max_length=6)
    十時単勝オッズ = models.CharField(max_length=6)
    十時複勝オッズ = models.CharField(max_length=6)
    コーナー順位1 = models.CharField(max_length=2)
    コーナー順位2 = models.CharField(max_length=2)
    コーナー順位3 = models.CharField(max_length=2)
    コーナー順位4 = models.CharField(max_length=2)
    前3F先頭差 = models.CharField(max_length=3)
    後3F先頭差 = models.CharField(max_length=3)
    騎手コード = models.CharField(max_length=5)
    調教師コード = models.CharField(max_length=5)
    馬体重 = models.CharField(max_length=3)
    馬体重増減 = models.CharField(max_length=3)
    天候コード = models.CharField(choices=天候コード, max_length=1)
    コース = models.CharField(max_length=1)
    脚質コード = models.CharField(choices=脚質コード, max_length=1)
    単勝 = models.CharField(max_length=7)
    複勝 = models.CharField(max_length=7)
    本賞金 = models.CharField(max_length=5)
    収得賞金 = models.CharField(max_length=5)
    レースペース流れ = models.CharField(max_length=2)
    馬ペース流れ = models.CharField(max_length=2)
    四角コース取り = models.CharField(max_length=1)
    予備2 = models.CharField(max_length=4)
