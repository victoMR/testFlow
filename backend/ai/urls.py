from django.urls import path
from . import views  # Cambiado de ...ai import views a . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('abrir-camara/', views.abrir_camara, name='abrir_camara'),
    path('procesar-pdf/', views.procesar_pdf, name='procesar_pdf'),
    path('procesar-imagen/', views.procesar_imagen, name='procesar_imagen'),
]