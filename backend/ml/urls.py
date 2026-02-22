from django.urls import path
from . import views

urlpatterns = [
    path('races/', views.RaceListView.as_view(), name='ml-races'),
    path('fields/', views.FieldListView.as_view(), name='ml-fields'),
    path('csv/download/', views.CsvDownloadView.as_view(), name='ml-csv'),
    path('ml/predict/', views.PredictView.as_view(), name='ml-predict'),
]
