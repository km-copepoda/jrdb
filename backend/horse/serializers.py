from rest_framework import serializers
from database.models.PreviousDayInformation import 馬基本情報, 前日_詳細情報

class HorseListSerializer(serializers.ModelSerializer):
    class Meta:
        model = 馬基本情報
        fields = ('血統登録番号', '馬名', '性別コード', '毛色コード', '生年月日')


class RaceDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = 前日_詳細情報
        fields = '__all__'


class HorseDetailSerializer(serializers.ModelSerializer):
    全レース_詳細情報 = RaceDetailSerializer(many=True)
    class Meta:
        model = 馬基本情報
        fields = ('血統登録番号', '馬名', '性別コード', '毛色コード', '生年月日', '馬主名', '生産者名', '産地名',
                 '父馬基本情報', '母馬基本情報', '母父馬基本情報', '全レース_詳細情報')

