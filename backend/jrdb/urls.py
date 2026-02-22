from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from rest_framework import routers
from horse.views import HorseViewSets

router = routers.DefaultRouter(trailing_slash=False)
router.register(r'horses', HorseViewSets)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/', include('ml.urls')),
    path('admin/', admin.site.urls),
    # React SPA のエンドポイント
    path('', TemplateView.as_view(template_name='index.html')),
]

