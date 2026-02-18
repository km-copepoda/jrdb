import itertools
from decimal import Decimal

from django.db import models
from .helpers import (
    場コード,
    天候コード,
    馬場状態コード,
    性別コード,
    毛色コード,
    種別コード,
    グレードコード,
    脚質コード,
    距離適性コード,
    調教矢印コード,
    厩舎評価コード,
    蹄コード,
    クラスコード,
    休養理由分類コード,
)


class 馬基本情報(models.Model):
    """
    file name is UKC
    """

    def 父馬基本情報(self):
        return 馬基本情報.objects.filter(馬名=self.父馬名).first()

    def 母馬基本情報(self):
        return 馬基本情報.objects.filter(馬名=self.母馬名).first()

    def 母父馬基本情報(self):
        return 馬基本情報.objects.filter(馬名=self.母父馬名).first()

    def 全レース_詳細情報(self):
        return 前日_詳細情報.objects.filter(馬基本情報=self)

    血統登録番号 = models.CharField(max_length=8, primary_key=True, db_index=True)
    馬名 = models.CharField(max_length=36)
    性別コード = models.CharField(choices=性別コード, max_length=1)
    毛色コード = models.CharField(choices=毛色コード, max_length=2)
    馬記号コード = models.CharField(max_length=2)
    父馬名 = models.CharField(max_length=36)
    母馬名 = models.CharField(max_length=36)
    母父馬名 = models.CharField(max_length=36)
    生年月日 = models.CharField(max_length=8)
    父馬生年 = models.CharField(max_length=4)
    母馬生年 = models.CharField(max_length=4)
    母父馬生年 = models.CharField(max_length=4)
    馬主名 = models.CharField(max_length=40)
    馬主会コード = models.CharField(max_length=2)
    生産者名 = models.CharField(max_length=40)
    産地名 = models.CharField(max_length=8)
    登録抹消フラグ = models.CharField(max_length=1)
    データ年月日 = models.CharField(max_length=8)
    父系統コード = models.CharField(max_length=4)
    母父系統コード = models.CharField(max_length=4)
    予備 = models.CharField(max_length=6)


class 前日_調教師情報(models.Model):
    """
    file name is CZA
    """

    調教師コード = models.CharField(max_length=5, primary_key=True)
    登録抹消フラグ = models.CharField(max_length=1)
    登録抹消年月日 = models.CharField(max_length=8)
    調教師名 = models.CharField(max_length=12)
    調教師カナ = models.CharField(max_length=30)
    調教師名略称 = models.CharField(max_length=6)
    所属コード = models.CharField(max_length=1)
    所属地域名 = models.CharField(max_length=4)
    生年月日 = models.CharField(max_length=8)
    初免許年 = models.CharField(max_length=4)
    調教師コメント = models.CharField(max_length=40)
    コメント入力年月日 = models.CharField(max_length=8)
    本年リーディング = models.CharField(max_length=3)
    本年平地成績 = models.CharField(max_length=12)
    本年障害成績 = models.CharField(max_length=12)
    本年特別勝数 = models.CharField(max_length=3)
    本年重賞勝数 = models.CharField(max_length=3)
    昨年リーディング = models.CharField(max_length=3)
    昨年平地成績 = models.CharField(max_length=12)
    昨年障害成績 = models.CharField(max_length=12)
    昨年特別勝数 = models.CharField(max_length=3)
    昨年重賞勝数 = models.CharField(max_length=3)
    通算平地成績 = models.CharField(max_length=20)
    通算障害成績 = models.CharField(max_length=20)
    データ年月日 = models.CharField(max_length=8)
    予備 = models.CharField(max_length=29)


class マスタ_騎手データ(models.Model):
    """
    file name is KZA
    """

    騎手コード = models.CharField(max_length=5, primary_key=True)
    登録抹消フラグ = models.CharField(max_length=1)
    登録抹消年月日 = models.CharField(max_length=8)
    騎手名 = models.CharField(max_length=12)
    騎手カナ = models.CharField(max_length=30)
    騎手名略称 = models.CharField(max_length=6)
    所属コード = models.CharField(max_length=1)
    所属地域名 = models.CharField(max_length=4)
    生年月日 = models.CharField(max_length=8)
    初免許年 = models.CharField(max_length=4)
    見習い区分 = models.CharField(max_length=1)
    所属厩舎 = models.CharField(max_length=5)
    騎手コメント = models.CharField(max_length=40)
    コメント入力年月日 = models.CharField(max_length=8)
    本年リーディング = models.CharField(max_length=3)
    本年平地成績 = models.CharField(max_length=12)
    本年障害成績 = models.CharField(max_length=12)
    本年特別勝数 = models.CharField(max_length=3)
    本年重賞勝数 = models.CharField(max_length=3)
    昨年リーディング = models.CharField(max_length=3)
    昨年平地成績 = models.CharField(max_length=12)
    昨年障害成績 = models.CharField(max_length=12)
    昨年特別勝数 = models.CharField(max_length=3)
    昨年重賞勝数 = models.CharField(max_length=3)
    通算平地成績 = models.CharField(max_length=20)
    通算障害成績 = models.CharField(max_length=20)
    データ年月日 = models.CharField(max_length=8)
    予備 = models.CharField(max_length=23)


class 前日_開催情報(models.Model):
    """
    file name is KAB
    """

    開催情報ID = models.CharField(max_length=6, primary_key=True)
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    年月日 = models.CharField(max_length=8)
    開催区分 = models.CharField(max_length=1)
    曜日 = models.CharField(max_length=2)
    場名 = models.CharField(max_length=4)
    天候コード = models.CharField(choices=天候コード, max_length=1)
    芝馬場状態コード = models.CharField(max_length=2)
    芝馬場状態内 = models.CharField(max_length=1)
    芝馬場状態中 = models.CharField(max_length=1)
    芝馬場状態外 = models.CharField(max_length=1)
    芝馬場差 = models.CharField(max_length=3)
    直線馬場差最内 = models.CharField(max_length=2)
    直線馬場差内 = models.CharField(max_length=2)
    直線馬場差中 = models.CharField(max_length=2)
    直線馬場差外 = models.CharField(max_length=2)
    直線馬場差大外 = models.CharField(max_length=2)
    ダ馬場状態コード = models.CharField(max_length=2)
    ダ馬場状態内 = models.CharField(max_length=1)
    ダ馬場状態中 = models.CharField(max_length=1)
    ダ馬場状態外 = models.CharField(max_length=1)
    ダ馬場差 = models.CharField(max_length=3)
    データ区分 = models.CharField(max_length=1)
    連続何日目 = models.CharField(max_length=2)
    芝種類 = models.CharField(max_length=1)
    草丈 = models.CharField(max_length=4)
    転圧 = models.CharField(max_length=1)
    凍結防止剤 = models.CharField(max_length=1)
    中間降水量 = models.CharField(max_length=5)
    予備 = models.CharField(max_length=7)


class 前日_番組情報(models.Model):
    """
    file name is BAC
    """

    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_番組情報-前日_開催情報_IDX")
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    番組情報ID = models.CharField(max_length=8, primary_key=True)
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    年月日 = models.CharField(max_length=8)
    発走時間 = models.CharField(max_length=4)
    距離 = models.CharField(max_length=4)
    芝ダ障害コード = models.CharField(max_length=1)
    右左 = models.CharField(max_length=1)
    内外 = models.CharField(max_length=1)
    種別コード = models.CharField(choices=種別コード, max_length=2)
    条件 = models.CharField(max_length=2)
    記号 = models.CharField(max_length=3)
    重量 = models.CharField(max_length=1)
    グレードコード = models.CharField(choices=グレードコード, max_length=1)
    レース名 = models.CharField(max_length=50)
    回数 = models.CharField(max_length=8)
    頭数 = models.CharField(max_length=2)
    コース = models.CharField(max_length=1)
    開催区分 = models.CharField(max_length=1)
    レース名短縮 = models.CharField(max_length=8)
    レース名9文字 = models.CharField(max_length=18)
    データ区分 = models.CharField(max_length=1)
    一着賞金 = models.CharField(max_length=5)
    二着賞金 = models.CharField(max_length=5)
    三着賞金 = models.CharField(max_length=5)
    四着賞金 = models.CharField(max_length=5)
    五着賞金 = models.CharField(max_length=5)
    一着算入賞金 = models.CharField(max_length=5)
    二着算入賞金 = models.CharField(max_length=5)
    単勝 = models.CharField(max_length=1)
    複勝 = models.CharField(max_length=1)
    枠連 = models.CharField(max_length=1)
    馬連 = models.CharField(max_length=1)
    馬単 = models.CharField(max_length=1)
    ワイド = models.CharField(max_length=1)
    三連複 = models.CharField(max_length=1)
    三連単 = models.CharField(max_length=1)
    馬券発売フラグ予備 = models.CharField(max_length=8)
    WIN5フラグ = models.CharField(max_length=1)
    WIN5フラグ予備 = models.CharField(max_length=5)


class 前日_基準三連単情報(models.Model):
    """
    file name is OV
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_基準三連単情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_基準三連単情報-前日_番組情報_IDX")
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
    登録頭数 = models.CharField(max_length=2)
    三連単オッズ = models.TextField(max_length=34272)
    予備 = models.CharField(max_length=4)

    def 三連単オッズ_to_dict(self):
        byte = 7
        max_horse = 18
        odds = {}
        for index, id in enumerate(
            list(itertools.permutations(range(1, max_horse + 1), 3))
        ):
            offset = index * byte
            value = self.三連単オッズ[offset : offset + byte].strip()
            if not value:
                continue
            decimal_calc = float(Decimal(value) * Decimal("0.1"))
            odds[f"{id[0]}-{id[1]}-{id[2]}"] = (
                decimal_calc if decimal_calc <= 999999 else -1
            )
        return odds


class 前日_基準ワイド情報(models.Model):
    """
    file name is OW
    """

    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_基準ワイド情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_基準ワイド情報-前日_番組情報_IDX")
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
    登録頭数 = models.CharField(max_length=2)
    ワイドオッズ = models.TextField(max_length=765)
    予備 = models.CharField(max_length=3)

    def ワイドオッズ_to_dict(self):
        byte = 5
        max_horse = 18
        odds = {}
        for index, id in enumerate(
            list(itertools.combinations(range(1, max_horse + 1), 2))
        ):
            offset = index * byte
            value = self.ワイドオッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[(id[0], id[1])] = float(value) if float(value) <= 999 else -1
        return odds


class 前日_基本馬単情報(models.Model):
    """
    file name is OU
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_基本馬単情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_基本馬単情報-前日_番組情報_IDX")
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
    登録頭数 = models.CharField(max_length=2)
    馬単オッズ = models.TextField(max_length=1836)
    予備 = models.CharField(max_length=8)

    def 馬単オッズ_to_dict(self):
        byte = 6
        max_horse = 18
        odds = {}
        for index, id in enumerate(
            list(itertools.permutations(range(1, max_horse + 1), 2))
        ):
            offset = index * byte
            value = self.馬単オッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[f"{id[0]}-{id[1]}"] = float(value) if float(value) <= 9999 else -1
        return odds


class 前日_基準三連複情報(models.Model):
    """
    file name is OT
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_基本三連複情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_基本三連複情報-前日_番組情報_IDX")
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
    登録頭数 = models.CharField(max_length=2)
    三連複オッズ = models.TextField(max_length=4896)
    予備 = models.CharField(max_length=4)

    def 三連複オッズ_to_dict(self):
        byte = 6
        max_horse = 18
        odds = {}
        for index, id in enumerate(
            list(itertools.combinations(range(1, max_horse + 1), 3))
        ):
            offset = index * byte
            value = self.三連複オッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[(id[0], id[1])] = float(value) if float(value) <= 9999 else -1
        return odds


class 前日_基準単複連情報(models.Model):
    """
    file name is OZ
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_基本単複連情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_基本単複連情報-前日_番組情報_IDX")
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
    登録頭数 = models.CharField(max_length=2)
    単勝オッズ = models.TextField(max_length=90)
    複勝オッズ = models.TextField(max_length=90)
    連勝オッズ = models.TextField(max_length=765)

    def 単勝オッズ_to_dict(self):
        byte = 5
        max_horse = 18
        odds = {}
        for index, id in enumerate(range(1, max_horse + 1)):
            offset = index * byte
            value = self.単勝オッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[str(id)] = float(value) if float(value) <= 999 else -1
        return odds

    def 複勝オッズ_to_dict(self):
        byte = 5
        max_horse = 18
        odds = {}
        for index, id in enumerate(range(1, max_horse + 1)):
            offset = index * byte
            value = self.複勝オッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[str(id)] = float(value) if float(value) <= 999 else -1
        return odds

    def 連勝オッズ_to_dict(self):
        byte = 5
        max_horse = 18
        odds = {}
        for index, id in enumerate(
            list(itertools.combinations(range(1, max_horse + 1), 2))
        ):
            offset = index * byte
            value = self.連勝オッズ[offset : offset + byte].strip()
            if not value:
                continue
            odds[(id[0], id[1])] = float(value) if float(value) <= 999 else -1
        return odds


class 前日_競走馬情報(models.Model):
    """
    file name is KYI
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_競走馬情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_競走馬情報-前日_番組情報_IDX"),
            models.Index(fields=["馬基本情報"], name="前日_競走馬情報-馬基本情報_IDX"),
            models.Index(fields=["マスタ_騎手データ"], name="前日_競走馬情報-マスタ_騎手データ_IDX"),
            models.Index(fields=["前日_調教師情報"], name="前日_競走馬情報-前日_調教師情報_IDX")
        ]

    前日_開催情報 = models.ForeignKey(前日_開催情報, on_delete=models.CASCADE, max_length=6, db_index=False)
    前日_番組情報 = models.ForeignKey(前日_番組情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    競走馬情報ID = models.CharField(max_length=10, primary_key=True)
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    馬番 = models.CharField(max_length=2)
    馬基本情報 = models.ForeignKey(馬基本情報, on_delete=models.CASCADE, max_length=8, db_index=False)
    馬名 = models.CharField(max_length=36)
    IDM = models.CharField(max_length=5)
    騎手指数 = models.CharField(max_length=5)
    情報指数 = models.CharField(max_length=5)
    予備1 = models.CharField(max_length=5)
    予備2 = models.CharField(max_length=5)
    予備3 = models.CharField(max_length=5)
    総合指数 = models.CharField(max_length=5)
    脚質コード = models.CharField(choices=脚質コード, max_length=1)
    距離適性コード = models.CharField(choices=距離適性コード, max_length=1)
    上昇度 = models.CharField(max_length=1)
    ローテーション = models.CharField(max_length=3)
    基準オッズ = models.CharField(max_length=5)
    基準人気順位 = models.CharField(max_length=2)
    基準複勝オッズ = models.CharField(max_length=5)
    基準複勝人気順位 = models.CharField(max_length=2)
    特定情報_本命 = models.CharField(max_length=3)
    特定情報_対抗 = models.CharField(max_length=3)
    特定情報_単穴 = models.CharField(max_length=3)
    特定情報_連下1 = models.CharField(max_length=3)
    特定情報_連下2 = models.CharField(max_length=3)
    総合情報_本命 = models.CharField(max_length=3)
    総合情報_対抗 = models.CharField(max_length=3)
    総合情報_単穴 = models.CharField(max_length=3)
    総合情報_連下1 = models.CharField(max_length=3)
    総合情報_連下2 = models.CharField(max_length=3)
    人気指数 = models.CharField(max_length=5)
    調教指数 = models.CharField(max_length=5)
    厩舎指数 = models.CharField(max_length=5)
    調教矢印コード = models.CharField(choices=調教矢印コード, max_length=1)
    厩舎評価コード = models.CharField(choices=厩舎評価コード, max_length=1)
    騎手期待連対率 = models.CharField(max_length=4)
    激走指数 = models.CharField(max_length=3)
    蹄コード = models.CharField(choices=蹄コード, max_length=2)
    重適正コード = models.CharField(max_length=1)
    クラスコード = models.CharField(choices=クラスコード, max_length=2)
    予備4 = models.CharField(max_length=2)
    ブリンカー = models.CharField(max_length=1)
    騎手名 = models.CharField(max_length=12)
    負担重量 = models.CharField(max_length=3)
    見習い区分 = models.CharField(max_length=1)
    調教師名 = models.CharField(max_length=12)
    調教師所属 = models.CharField(max_length=4)
    前走1競走成績キー = models.CharField(max_length=16)
    前走2競走成績キー = models.CharField(max_length=16)
    前走3競走成績キー = models.CharField(max_length=16)
    前走4競走成績キー = models.CharField(max_length=16)
    前走5競走成績キー = models.CharField(max_length=16)
    前走1レースキー = models.CharField(max_length=8)
    前走2レースキー = models.CharField(max_length=8)
    前走3レースキー = models.CharField(max_length=8)
    前走4レースキー = models.CharField(max_length=8)
    前走5レースキー = models.CharField(max_length=8)
    枠番 = models.CharField(max_length=1)
    予備5 = models.CharField(max_length=2)
    総合印 = models.CharField(max_length=1)
    IDM印 = models.CharField(max_length=1)
    情報印 = models.CharField(max_length=1)
    騎手印 = models.CharField(max_length=1)
    厩舎印 = models.CharField(max_length=1)
    調教印 = models.CharField(max_length=1)
    激走印 = models.CharField(max_length=1)
    芝適性コード = models.CharField(max_length=1)
    ダ適性コード = models.CharField(max_length=1)
    マスタ_騎手データ = models.ForeignKey(マスタ_騎手データ, on_delete=models.CASCADE, max_length=5, db_index=False)
    前日_調教師情報 = models.ForeignKey(前日_調教師情報, on_delete=models.CASCADE, max_length=5, db_index=False)
    予備6 = models.CharField(max_length=1)
    獲得賞金 = models.CharField(max_length=6)
    収得賞金 = models.CharField(max_length=5)
    条件クラス = models.CharField(max_length=1)
    テン指数 = models.CharField(max_length=5)
    ペース指数 = models.CharField(max_length=5)
    上がり指数 = models.CharField(max_length=5)
    位置指数 = models.CharField(max_length=5)
    ペース予想 = models.CharField(max_length=1)
    道中順位 = models.CharField(max_length=2)
    道中差 = models.CharField(max_length=2)
    道中内外 = models.CharField(max_length=1)
    後3F順位 = models.CharField(max_length=2)
    後3F差 = models.CharField(max_length=2)
    後3F内外 = models.CharField(max_length=1)
    ゴール順位 = models.CharField(max_length=2)
    ゴール差 = models.CharField(max_length=2)
    ゴール内外 = models.CharField(max_length=1)
    展開記号 = models.CharField(max_length=1)
    距離適性コード2 = models.CharField(max_length=1)
    枠確定馬体重 = models.CharField(max_length=3)
    枠確定馬体重増減 = models.CharField(max_length=3)
    取消フラグ = models.CharField(max_length=1)
    性別コード = models.CharField(choices=性別コード, max_length=1)
    馬主名 = models.CharField(max_length=40)
    馬主会コード = models.CharField(max_length=2)
    馬記号コード = models.CharField(max_length=2)
    激走順位 = models.CharField(max_length=2)
    LS指数順位 = models.CharField(max_length=2)
    テン指数順位 = models.CharField(max_length=2)
    ペース指数順位 = models.CharField(max_length=2)
    上がり指数順位 = models.CharField(max_length=2)
    位置指数順位 = models.CharField(max_length=2)
    騎手期待単勝率 = models.CharField(max_length=4)
    騎手期待3着内率 = models.CharField(max_length=4)
    輸送区分 = models.CharField(max_length=1)
    走法 = models.CharField(max_length=8)
    体型 = models.CharField(max_length=24)
    体型総合1 = models.CharField(max_length=3)
    体型総合2 = models.CharField(max_length=3)
    体型総合3 = models.CharField(max_length=3)
    馬特記1 = models.CharField(max_length=3)
    馬特記2 = models.CharField(max_length=3)
    馬特記3 = models.CharField(max_length=3)
    馬スタート指数 = models.CharField(max_length=4)
    馬出遅率 = models.CharField(max_length=4)
    参考前走 = models.CharField(max_length=2)
    参考前走騎手コード = models.CharField(max_length=5)
    万券指数 = models.CharField(max_length=3)
    万券印 = models.CharField(max_length=1)
    降級フラグ = models.CharField(max_length=1)
    激走タイプ = models.CharField(max_length=2)
    休養理由分類コード = models.CharField(休養理由分類コード, max_length=2)
    フラグ = models.CharField(max_length=16)
    入厩何走目 = models.CharField(max_length=2)
    入厩年月日 = models.CharField(max_length=8)
    入厩何日前 = models.CharField(max_length=3)
    放牧先 = models.CharField(max_length=50)
    放牧先ランク = models.CharField(max_length=1)
    厩舎ランク = models.CharField(max_length=1)
    予備7 = models.TextField(max_length=398)

    def 前走レース(self):
        if self.前走1レースキー:
            return 前日_競走馬情報.objects.filter(前日_番組情報=self.前走1レースキー)
        return []

    def 前前走レース(self):
        if self.前走2レースキー:
            return 前日_競走馬情報.objects.filter(前日_番組情報=self.前走2レースキー)
        return []

    def 前前前走レース(self):
        if self.前走3レースキー:
            return 前日_競走馬情報.objects.filter(前日_番組情報=self.前走3レースキー)
        return []

    def 前前前前走レース(self):
        if self.前走4レースキー:
            return 前日_競走馬情報.objects.filter(前日_番組情報=self.前走4レースキー)
        return []

    def 前前前前前走レース(self):
        if self.前走5レースキー:
            return 前日_競走馬情報.objects.filter(前日_番組情報=self.前走5レースキー)
        return []

    def 前走_競走馬情報(self):
        if self.前走1レースキー:
            return 前日_競走馬情報.objects.filter(
                前日_番組情報=self.前走1レースキー, 馬基本情報=self.馬基本情報
            ).first()
        return None

    def 前前走_競走馬情報(self):
        if self.前走2レースキー:
            return 前日_競走馬情報.objects.filter(
                前日_番組情報=self.前走2レースキー, 馬基本情報=self.馬基本情報
            ).first()
        return None

    def 前前前走_競走馬情報(self):
        if self.前走3レースキー:
            return 前日_競走馬情報.objects.filter(
                前日_番組情報=self.前走3レースキー, 馬基本情報=self.馬基本情報
            ).first()
        return None

    def 前前前前走4_競走馬情報(self):
        if self.前走4レースキー:
            return 前日_競走馬情報.objects.filter(
                前日_番組情報=self.前走4レースキー, 馬基本情報=self.馬基本情報
            ).first()
        return None

    def 前前前前前走5_競走馬情報(self):
        if self.前走5レースキー:
            return 前日_競走馬情報.objects.filter(
                前日_番組情報=self.前走5レースキー, 馬基本情報=self.馬基本情報
            ).first()
        return None


class 前日_調教分析情報(models.Model):
    """
    CYB
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_調教分析情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_調教分析情報-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="前日_調教分析情報-前日_競走馬情報_IDX"),
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
    調教タイプ = models.CharField(max_length=2)
    調教コース種別 = models.CharField(max_length=1)
    坂 = models.CharField(max_length=2)
    W = models.CharField(max_length=2)
    ダ = models.CharField(max_length=2)
    芝 = models.CharField(max_length=2)
    プ = models.CharField(max_length=2)
    障 = models.CharField(max_length=2)
    ポ = models.CharField(max_length=2)
    調教距離 = models.CharField(max_length=1)
    調教重点 = models.CharField(max_length=1)
    追切指数 = models.CharField(max_length=3)
    仕上指数 = models.CharField(max_length=3)
    調教量評価 = models.CharField(max_length=1)
    仕上指数変化 = models.CharField(max_length=1)
    調教コメント = models.CharField(max_length=40)
    コメント年月日 = models.CharField(max_length=8)
    調教評価 = models.CharField(max_length=1)
    予備 = models.CharField(max_length=8)


class 前日_調教本追切情報(models.Model):
    """
    file name is CHA
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_調教本追切情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_調教本追切情報-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="前日_調教本追切情報-前日_競走馬情報_IDX"),
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
    曜日 = models.CharField(max_length=2)
    調教年月日 = models.CharField(max_length=8)
    回数 = models.CharField(max_length=1)
    調教コースコード = models.CharField(max_length=2)
    追切種類 = models.CharField(max_length=1)
    追い状態 = models.CharField(max_length=2)
    乗り役 = models.CharField(max_length=1)
    調教F = models.CharField(max_length=1)
    テンF = models.CharField(max_length=3)
    中間F = models.CharField(max_length=3)
    終いF = models.CharField(max_length=3)
    テンF指数 = models.CharField(max_length=3)
    中間F指数 = models.CharField(max_length=3)
    終いF指数 = models.CharField(max_length=3)
    追切指数 = models.CharField(max_length=3)
    相手_併せ結果 = models.CharField(max_length=1)
    相手_追切種類 = models.CharField(max_length=1)
    相手_年齢 = models.CharField(max_length=2)
    相手_クラス = models.CharField(max_length=2)
    予備 = models.CharField(max_length=7)


class 前日_競走馬拡張(models.Model):
    """
    file name is KKA
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_競走馬拡張-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_競走馬拡張-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="前日_競走馬拡張-前日_競走馬情報_IDX"),
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
    JRA成績 = models.CharField(max_length=3)
    交流成績 = models.CharField(max_length=3)
    他成績 = models.CharField(max_length=3)
    芝ダ障害別成績 = models.CharField(max_length=3)
    芝ダ障害別距離成績 = models.CharField(max_length=3)
    トラック距離成績 = models.CharField(max_length=3)
    ローテ成績 = models.CharField(max_length=3)
    回り成績 = models.CharField(max_length=3)
    騎手成績 = models.CharField(max_length=3)
    良成績 = models.CharField(max_length=3)
    稍成績 = models.CharField(max_length=3)
    重成績 = models.CharField(max_length=3)
    Sペース成績 = models.CharField(max_length=3)
    Mペース成績 = models.CharField(max_length=3)
    Hペース成績 = models.CharField(max_length=3)
    季節成績 = models.CharField(max_length=3)
    枠成績 = models.CharField(max_length=3)
    騎手距離成績 = models.CharField(max_length=3)
    騎手トラック距離成績 = models.CharField(max_length=3)
    騎手調教師別成績 = models.CharField(max_length=3)
    騎手馬主別成績 = models.CharField(max_length=3)
    騎手ブリンカ成績 = models.CharField(max_length=3)
    調教師馬主別成績 = models.CharField(max_length=3)
    父馬産駒芝連対率 = models.CharField(max_length=3)
    父馬産駒ダ連対率 = models.CharField(max_length=3)
    父馬産駒連対平均距離 = models.CharField(max_length=4)
    母父馬産駒芝連対率 = models.CharField(max_length=3)
    母父馬産駒ダ連対率 = models.CharField(max_length=3)
    母父馬産駒連対平均距離 = models.CharField(max_length=4)
    予備 = models.CharField(max_length=16)


class 前日_詳細情報(models.Model):
    """
    file name is JOA
    """
    class Meta:
        indexes = [
            models.Index(fields=["前日_開催情報"], name="前日_詳細情報-前日_開催情報_IDX"),
            models.Index(fields=["前日_番組情報"], name="前日_詳細情報-前日_番組情報_IDX"),
            models.Index(fields=["前日_競走馬情報"], name="前日_詳細情報-前日_競走馬情報_IDX"),
            models.Index(fields=["馬基本情報"], name="前日_詳細情報-馬基本情報_IDX"),
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
    馬名 = models.CharField(max_length=36)
    基準オッズ = models.CharField(max_length=5)
    基準複勝オッズ = models.CharField(max_length=5)
    CID調教素点 = models.CharField(max_length=5)
    CID厩舎素点 = models.CharField(max_length=5)
    CID素点 = models.CharField(max_length=5)
    CID = models.CharField(max_length=3)
    LS指数 = models.CharField(max_length=5)
    LS評価 = models.CharField(max_length=1)
    EM = models.CharField(max_length=1)
    厩舎BB印 = models.CharField(max_length=1)
    厩舎BB_単勝回収率 = models.CharField(max_length=5)
    厩舎BB_連対率 = models.CharField(max_length=5)
    騎手BB印 = models.CharField(max_length=1)
    騎手BB_単勝回収率 = models.CharField(max_length=5)
    騎手BB_連対率 = models.CharField(max_length=5)
    予備 = models.CharField(max_length=3)
