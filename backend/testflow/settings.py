from pathlib import Path

# Construcción de la ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Clave secreta (mantener segura en producción)
SECRET_KEY = "django-insecure-2)o-9bd&_88mpi@c*0g_bs2)24efx%4vz%%ttx=y&6=+wz__@v"

# Debug debe ser False en producción
DEBUG = True

# Define los hosts permitidos para el despliegue
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# Definir aplicaciones instaladas
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ai",  # Incluye tu aplicación
    "corsheaders",  # Incluye CORS headers si usas frontend separado
]

# Middlewares
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",  # Middleware para CORS
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Configuración CORS (opcional)
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Agregar la URL de tu frontend si es diferente
]

# URL principal del proyecto
ROOT_URLCONF = "testflow.urls"

# Configuración de plantillas
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],  # Si tienes directorios de plantillas personalizadas, agrégalas aquí
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Aplicación WSGI
WSGI_APPLICATION = "testflow.wsgi.application"

# Configuración de la base de datos
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Validación de contraseñas
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internacionalización y zona horaria
LANGUAGE_CODE = "es-mx"
TIME_ZONE = "America/Mexico_City"
USE_I18N = True
USE_TZ = True

# Configuración de archivos estáticos y media
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]  # Carpeta de archivos estáticos
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"  # Carpeta para los archivos subidos

# Configuración para el ID predeterminado de los modelos
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
