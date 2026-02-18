from django.urls import path
from . import views

urlpatterns = {
    path('races/', views.RaceListView.as_view(), name='ml-races'),
    path('fields/', views.FieldListView.as_view(), name='ml-fields'),
    path('csv/', views.CsvDownloadVIew.as_view(), name='ml-csv'),
}