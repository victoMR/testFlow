from django.urls import path
from . import views

urlpatterns = [
    path("api/procesar_fotograma/", views.procesar_fotograma),
    path("api/procesar_pdf/", views.procesar_pdf),
    path("api/procesar_imagen/", views.procesar_imagen),
    path("api/procesar_texto/", views.procesar_texto),
]
