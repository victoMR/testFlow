#!/bin/bash
# Script de inicialización para despliegue automático
set -euo pipefail
IFS=$'\n\t'

# Variables de entorno
REPO_URL="https://github.com/victoMR/testFlow.git"
PROJECT_DIR="/home/ubuntu/testFlow"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="/var/log/userdata.log"

# Función para logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Crear directorio de logs
sudo mkdir -p "$(dirname "$LOG_FILE")"
sudo touch "$LOG_FILE"
sudo chown ubuntu:ubuntu "$LOG_FILE"

log "Iniciando script de despliegue..."

# Actualizar sistema
log "Actualizando sistema..."
sudo apt-get update && sudo apt-get upgrade -y

# Instalar dependencias básicas
log "Instalando dependencias básicas..."
sudo apt-get install -y python3-pip python3-venv git nginx

# Clonar repositorio
log "Clonando repositorio..."
if [ ! -d "$PROJECT_DIR" ]; then
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    log "El directorio del proyecto ya existe"
fi

# Crear archivo .env con las credenciales directamente
log "Configurando variables de entorno..."
cat > "$BACKEND_DIR/.env" << EOL
# Django Settings
DEBUG=False
SECRET_KEY='django-insecure-2)o-9bd&_88mpi@c*0g_bs2)24efx%4vz%%ttx=y&6=+wz__@v'

# MongoDB Settings
MONGO_URI='mongodb+srv://admin:L3J52w7zP9raHPel@testflowclouster.wxv1v.mongodb.net/?retryWrites=true&w=majority&appName=TestFlowClouster&ssl=true'
MONGO_USER='admin'
MONGO_PASSWORD='L3J52w7zP9raHPel'

# Redis Settings
REDIS_HOST='redis-11043.c274.us-east-1-3.ec2.redns.redis-cloud.com'
REDIS_PORT=11043
REDIS_PASSWORD='UYsHawLLBx6yeYoBSMQNZlXlSsBtnSIm'

# Allowed Hosts y CORS
ALLOWED_HOSTS=.amazonaws.com,localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
EOL

# Asegurar el archivo .env
sudo chown ubuntu:www-data "$BACKEND_DIR/.env"
sudo chmod 640 "$BACKEND_DIR/.env"

# Crear y activar entorno virtual
log "Configurando entorno virtual..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Instalar dependencias de Python
log "Instalando dependencias de Python..."
cd "$BACKEND_DIR"
pip install --upgrade pip
pip install -r requirements.txt

# Configurar y iniciar servicios
log "Configurando servicios..."

# Copiar archivo de servicio de uvicorn
sudo cp gunicorn/gunicorn.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gunicorn
sudo systemctl start gunicorn

# Verificar estado del servicio
if sudo systemctl is-active --quiet gunicorn; then
    log "Servicio gunicorn iniciado correctamente"
else
    log "Error: El servicio gunicorn no pudo iniciarse"
    exit 1
fi

# Configurar permisos
log "Configurando permisos..."
sudo chown -R ubuntu:www-data "$BACKEND_DIR"
sudo chmod -R 750 "$BACKEND_DIR"

log "Despliegue completado exitosamente"