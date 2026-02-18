from django.shortcuts import get_object_or_404
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from horse.serializers import HorseListSerializer, HorseDetailSerializer
from horse.filters import HorseFilter
from database.models.PreviousDayInformation import 馬基本情報

class HorseViewSets(ModelViewSet):
    queryset = 馬基本情報.objects.all().order_by('馬名')
    serializer_class = HorseListSerializer
    http_method_names = ['get']
    filter_class = HorseFilter

#    def list(self, request):
#        queryset = self.queryset
#        page = self.paginate_queryset(queryset)
#        if page is not None:
#            serializer = self.get_serializer(page, many=True)
#            return self.get_paginated_response(serializer.data)
#        serializer = self.get_serializer(queryset, many=True)
#        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        horse = get_object_or_404(self.queryset, pk=pk)
        serializer = HorseDetailSerializer(horse)
        return Response(serializer.data)

