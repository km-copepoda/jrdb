from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from horse.views import HorseViewSets

router = routers.DefaultRouter(trailing_slash=False)
router.register(r'horses', HorseViewSets)

urlpatterns = [
    path('api/', include(router.urls)),
    path('admin/', admin.site.urls),
]

