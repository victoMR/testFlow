from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("process-frame", views.procesar_fotograma, name="procesar_fotograma"),  # Sin barra al final y sin 'api/'
    path("procesar-pdf/", views.procesar_pdf, name="procesar_pdf"),
    path("procesar-imagen/", views.procesar_imagen, name="procesar_imagen"),
]
