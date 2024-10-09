from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    # Estas rutas ya no son necesarias aquí, ya que las manejamos en asgi.py
    path("procesar_fotograma/", views.procesar_fotograma, name="procesar_fotograma"),
    path("procesar_pdf/", views.procesar_pdf, name="procesar_pdf"),
    path("procesar_imagen/", views.procesar_imagen, name="procesar_imagen"),
    path("procesar_texto/", views.procesar_texto, name="procesar_texto")
]
