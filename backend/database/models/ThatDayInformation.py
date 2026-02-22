from django.db import models

from database.models.PreviousDayInformation import 前日_競走馬情報
from .helpers import 場コード, 馬場状態コード, 天候コード


class 直前_情報(models.Model):
    """
    file name is TYB
    """

    開催情報ID = models.CharField(max_length=6, db_index=True)
    番組情報ID = models.CharField(max_length=8, db_index=True)
    前日_競走馬情報 = models.OneToOneField(
        前日_競走馬情報, on_delete=models.CASCADE, max_length=10, primary_key=True
    )
    場コード = models.CharField(choices=場コード, max_length=2)
    年 = models.CharField(max_length=2)
    回 = models.CharField(max_length=1)
    日 = models.CharField(max_length=1)
    R = models.CharField(max_length=2)
    馬番 = models.CharField(max_length=2)
    IDM = models.CharField(max_length=5)
    騎手指数 = models.CharField(max_length=5)
    情報指数 = models.CharField(max_length=5)
    オッズ指数 = models.CharField(max_length=5)
    パドック指数 = models.CharField(max_length=5)
    予備1 = models.CharField(max_length=5)
    総合指数 = models.CharField(max_length=5)
    馬具変更情報 = models.CharField(max_length=1)
    脚元情報 = models.CharField(max_length=1)
    取消フラグ = models.CharField(max_length=1)
    騎手コード = models.CharField(max_length=5)
    騎手名 = models.CharField(max_length=12)
    負担重量 = models.CharField(max_length=3)
    見習い区分 = models.CharField(max_length=1)
    馬場状態コード = models.CharField(choices=馬場状態コード, max_length=2)
    天候コード = models.CharField(choices=天候コード, max_length=1)
    単勝オッズ = models.CharField(max_length=6)
    複勝オッズ = models.CharField(max_length=6)
    オッズ取得時間 = models.CharField(max_length=4)
    馬体重 = models.CharField(max_length=3)
    馬体重増減 = models.CharField(max_length=3)
    オッズ印 = models.CharField(max_length=1)
    パドック印 = models.CharField(max_length=1)
    直前総合印 = models.CharField(max_length=1)
    予備2 = models.CharField(max_length=29)
