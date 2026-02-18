import csv
from django.http import JsonResponse, StreamingHttpResponse
from django.views import views
from django.db.models import Exists, OuterRef

from database.models PreviousDayInformation import 前日_番組情報
from database.models.GradeInformation import 成績_成績分析用情報

# 着順・確定単勝人気順位の上位3位
TOP3_VALUES = ['1', '2', '3', '01', '02', '03', ' 1', ' 2', ' 3']

SED_ML_FIELDS = [
    'IDM', '素点', 'テン指数', '上がり指数', 'ペース指数',
    '確定単勝オッズ', '確定単勝人気順位', '馬体重', '馬体重増減',
    'コーナー順位1', 'コーナー順位2', 'コーナー順位3', 'コーナー順位4',
    '着順', '脚質コード', '馬ペース', 'レースペース',
]

KYI_ML_FIELDS = [
    'IDM', '騎手指数', '情報指数', '総合指数', '人気指数',
    '調教師数', '兵舎指数', '基準オッズ', '基準人気順位',
    '基準複勝オッズ', '基準複勝人気順位', 'テン指数', 'ペース指数',
    '上がり指数', '位置指数',
]

BAC_ML_FIELDS = [
    '距離', '芝ダ障害コード', 'グレードコード', '条件', '種別コード',
    '重量', '頭数', '年月日', '発走時間',
]

KAB_ML_FIELDS = [
    '天候コード', '芝馬場状態コード', 'ダ馬場状態コード',
    '芝馬場差', 'ダ馬場差',
]

ALL_FIELDS = {
    'SED': SED_ML_FIELDS,
    'KYI': KYI_ML_FIELDS,
    'BAC': BAC_ML_FIELDS,
    'KAB': KAB_ML_FIELDS,
}


def _favs_in_top3_subquery():
    """
    EXISTS サブクエリ：あるレースで確定単勝人気順位1-3の馬が着順1-3に入っているか。
    荒れた = NOT EXISTS(このサブクエリ)
    """
    return Exists(
        成績_成績分析用情報.objects.filter(
            前日_番組情報=OuterRef('pk'),
            確定単勝人気順位__in=TOP3_VALUES,
            着順__in=TOP3_VALUES,
        )
    )


def _has_results_subquery():
    """EXISTS サブクエリ：成績データ(SED)が存在するレースかどうか"""
    return Exists(
        成績_成績分析用情報.objects.filter(
            前日_番組情報=OuterRef('pk'),
        )
    )

class RaceListView(View):
    """
    GET /api/ml/races/
    クエリパラメータ：
        date_from / date_to  - 年月日(YYYYMMDD)
        venue                - 場コード (01-10)
        grade                - グレードコード(1-6)
        track                - 芝ダ障害コード(1=芝, 2=ダ, 3=障害)
        distance_min / distance_max - 距離(数値文字列)
        areta_only           - 1にすると荒れたレースのみ
        page / page_size     - ページネーション
    """

    def get(self, request):
        pass


class FieldListView(View):
    """
    GET /api/ml/fields/
    MLに使える選択可能フィールド一覧をモデル別に返す
    """
    
    def get(self, request):
        return JsonResponse({
            model: [{'key': f, 'label': f} for f in fields]
            for model, fields in ALL_FIELDS.items()
        })

class _EchoBuffer:
    """StreamingHttpResponse用の疑似バッファ"""
    def write(self, value):
        return value

class CsvDownloadView(View):
    """
    GET /api/ml/csv/?races=id1,id2&fields=SED.IDM,KYI.騎手指数,...
    CSV 1行 = 馬1頭分 StreamingHttpResponseで大量データに対応
    """

    def get(self, request):
        pass