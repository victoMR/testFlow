"""
ASGI config for testflow project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path, include
from ai import views  # Importa tus vistas aquí

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "testflow.settings")

django_asgi_app = get_asgi_application()

# Definir las rutas de la API
api_routes = [
    path("api/procesar_fotograma/", views.procesar_fotograma),
    path("api/procesar_pdf/", views.procesar_pdf),
    path("api/procesar_imagen/", views.procesar_imagen),
    path("api/procesar_texto/", views.procesar_texto),
]

application = ProtocolTypeRouter(
    {
        "http": URLRouter(
            api_routes
            + [path("", django_asgi_app)]
        ),
    }
)

# Imprime las rutas para depuración
for route in api_routes:
    print(f"Ruta registrada: {route.pattern}")
