import csv
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.db.models import Exists, OuterRef

from database.models.PreviousDayInformation import (
    前日_番組情報,
    前日_競走馬情報,
    前日_開催情報,
)
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
    '調教指数', '厩舎指数', '基準オッズ', '基準人気順位',
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

# モデル名→ORMモデル / レースとの結合方法のマッピング
MODEL_MAP = {
    'SED': {
        'model': 成績_成績分析用情報,
        'race_fk': '前日_番組情報',
    },
    'KYI': {
        'model': 前日_競走馬情報,
        'race_fk': '前日_番組情報',
    },
    'BAC': {
        'model': 前日_番組情報,
        'race_fk': None,  # BAC自身がレース
    },
    'KAB': {
        'model': 前日_開催情報,
        'race_fk': None,  # 開催情報はレースの親
    },
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
        qs = 前日_番組情報.objects.annotate(
            has_results=_has_results_subquery(),
            favs_in_top3=_favs_in_top3_subquery(),
        ).filter(has_results=True).order_by('-年月日', '場コード', 'R')

        # フィルタ適用
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')
        if date_from:
            qs = qs.filter(年月日__gte=date_from)
        if date_to:
            qs = qs.filter(年月日__lte=date_to)

        venue = request.GET.get('venue')
        if venue:
            qs = qs.filter(場コード=venue)

        grade = request.GET.get('grade')
        if grade:
            qs = qs.filter(グレードコード=grade)

        track = request.GET.get('track')
        if track:
            qs = qs.filter(芝ダ障害コード=track)

        distance_min = request.GET.get('distance_min')
        distance_max = request.GET.get('distance_max')
        if distance_min:
            qs = qs.filter(距離__gte=distance_min)
        if distance_max:
            qs = qs.filter(距離__lte=distance_max)

        areta_only = request.GET.get('areta_only')
        if areta_only == '1':
            qs = qs.filter(favs_in_top3=False)

        # ページネーション
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 50))
        total = qs.count()
        start = (page - 1) * page_size
        races = qs[start:start + page_size]

        results = []
        for r in races:
            results.append({
                'id': r.番組情報ID,
                'date': r.年月日,
                'venue': r.場コード,
                'race_num': r.R,
                'distance': r.距離,
                'track': r.芝ダ障害コード,
                'grade': r.グレードコード,
                'horse_count': r.頭数,
                'race_name': r.レース名.strip(),
                'areta': not r.favs_in_top3,
            })

        return JsonResponse({
            'results': results,
            'count': total,
        })


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
        race_ids_raw = request.GET.get('races', '')
        fields_raw = request.GET.get('fields', '')

        if not race_ids_raw or not fields_raw:
            return JsonResponse({'error': 'races と fields は必須です'}, status=400)

        race_ids = [rid.strip() for rid in race_ids_raw.split(',') if rid.strip()]
        field_specs = [f.strip() for f in fields_raw.split(',') if f.strip()]

        # フィールドをモデル別に分類 & バリデーション
        fields_by_model = {}
        header_columns = []
        for spec in field_specs:
            if '.' not in spec:
                return JsonResponse({'error': f'不正なフィールド形式: {spec}'}, status=400)
            model_name, field_name = spec.split('.', 1)
            if model_name not in ALL_FIELDS:
                return JsonResponse({'error': f'不明なモデル: {model_name}'}, status=400)
            if field_name not in ALL_FIELDS[model_name]:
                return JsonResponse({'error': f'不明なフィールド: {spec}'}, status=400)
            fields_by_model.setdefault(model_name, []).append(field_name)
            header_columns.append(spec)

        # SED (馬単位の基本)をベースにする - 馬1頭=1行
        sed_entries = 成績_成績分析用情報.objects.filter(
            前日_番組情報__in=race_ids,
        ).select_related(
            '前日_番組情報',
            '前日_番組情報__前日_開催情報',
            '前日_競走馬情報',
        ).order_by('前日_番組情報__年月日', '前日_番組情報__場コード', '前日_番組情報__R', '馬番')

        def generate_csv():
            buf = _EchoBuffer()
            writer = csv.writer(buf)

            # ヘッダー行
            yield writer.writerow(['race_id', '年月日', '馬番'] + header_columns)

            for entry in sed_entries.iterator():
                race = entry.前日_番組情報
                kab = race.前日_開催情報

                row = [race.番組情報ID, entry.年月日, entry.馬番]

                for col in header_columns:
                    model_name, field_name = col.split('.', 1)
                    if model_name == 'SED':
                        row.append(getattr(entry, field_name, ''))
                    elif model_name == 'KYI':
                        # KYI は前日_競走馬情報 (同じ馬・同じレース)
                        kyi = entry.前日_競走馬情報
                        row.append(getattr(kyi, field_name, ''))
                    elif model_name == 'BAC':
                        row.append(getattr(race, field_name, ''))
                    elif model_name == 'KAB':
                        row.append(getattr(kab, field_name, ''))

                yield writer.writerow(row)

        response = StreamingHttpResponse(generate_csv(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="jrdb_ml_data.csv"'
        return response
