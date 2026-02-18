from django_filters import rest_framework as filters
from database.models.PreviousDayInformation import 馬基本情報

class HorseFilter(filters.FilterSet):
    馬名 = filters.CharFilter(lookup_expr='contains')
    class Meta:
        model = 馬基本情報
        fields = ('馬名',)
