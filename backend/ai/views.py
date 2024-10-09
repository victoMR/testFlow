from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from .model import (
    Classifier,
    render_latex_to_image,
    process_with_im2latex,
    convertir_a_latex,
)
import os
import logging
import cv2
import base64
import io
import time
import asyncio
from channels.http import AsgiHandler
from django.core.handlers.asgi import ASGIRequest
import numpy as np
import hashlib
import redis
import json
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",  # Asegura que los logs puedan manejar caracteres Unicode
)
logger = logging.getLogger(__name__)

# Configuración de MongoDB
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    logger.error("MONGO_URI no está configurado en las variables de entorno")
    raise ValueError("MONGO_URI no está configurado")

try:
    client = MongoClient(mongo_uri)
    db = client["math_problems_db"]
    collection = db["problemas"]
    logger.info(f"Conexión exitosa a MongoDB en la uri")
except errors.ConnectionFailure as e:
    logger.error(f"No se pudo conectar a MongoDB: {e}")
    raise

# Configuración de Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD")
redis_db = int(os.getenv("REDIS_DB", 0))

# Configuración de Redis para caché
redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    db=redis_db,
    decode_responses=True,  # Esto asegura que las respuestas de Redis sean strings
)
cache = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    db=redis_db,
    decode_responses=True,
)
logger.info(f"Conexión exitosa a Redis en {redis_host}:{redis_port}/{redis_db}")

# Crear un ThreadPoolExecutor global
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)


async def index(scope, receive, send):
    await AsgiHandler()(scope, receive, send)


async def procesar_fotograma(scope, receive, send):
    if scope["method"] == "OPTIONS":
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"POST, OPTIONS"),
                    (b"access-control-allow-headers", b"Content-Type"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})
        return

    if scope["method"] != "POST":
        await send(
            {
                "type": "http.response.start",
                "status": 405,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"Method Not Allowed"})
        return

    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    if not body:
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": "No se proporcionó un fotograma".encode(),
            }
        )
        return

    try:
        img_hash = hashlib.md5(body).hexdigest()
        logger.info(f"Hash de la imagen calculado: {img_hash}")

        cached_result = redis_client.get(img_hash)
        if cached_result:
            logger.info(f"Resultado obtenido de caché para {img_hash}")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": cached_result.encode()})
            return

        nparr = np.frombuffer(body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen")

        processed_result = await asyncio.to_thread(process_with_im2latex, img)
        if processed_result is None:
            raise ValueError("Error al procesar la imagen con Im2Latex")

        # Asumimos que process_with_im2latex ahora devuelve una tupla (formula, tipo)
        cleaned_formula, problem_type = processed_result

        logger.info(
            f"Formula clasificada: {cleaned_formula}, Tipo de problema: {problem_type}"
        )

        if cleaned_formula:
            latex_image = await asyncio.to_thread(
                render_latex_to_image, cleaned_formula
            )
            buffered = io.BytesIO()
            latex_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            result = await asyncio.to_thread(
                collection.update_one,
                {"formula": cleaned_formula},
                {
                    "$setOnInsert": {
                        "formula": cleaned_formula,
                        "tipo": problem_type,
                        "usos": 0,
                    },
                    "$inc": {"usos": 1},
                },
                upsert=True,
            )

            message = (
                "Nuevo problema detectado y agregado a la base de datos."
                if result.upserted_id
                else "Problema existente actualizado en la base de datos."
            )

            response_data = {
                "formula": cleaned_formula,
                "tipo": problem_type,
                "message": message,
                "latex_image": img_str,
            }

            json_response = json.dumps(response_data)

            redis_client.setex(img_hash, settings.CACHE_TIMEOUT, json_response)

            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": json_response.encode()})
            return

        logger.warning("No se detectó ninguna fórmula en el fotograma.")
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": "No se detectó ninguna fórmula en el fotograma".encode(),
            }
        )

    except Exception as e:
        logger.error(f"Error al procesar el fotograma: {str(e)}", exc_info=True)
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": f"Error interno del servidor al procesar el fotograma: {str(e)}".encode(),
            }
        )


async def procesar_pdf(scope, receive, send):
    """Procesa un archivo PDF subido y extrae ecuaciones matemáticas."""
    request = ASGIRequest(scope, receive)

    if "pdf" not in request.FILES:
        response = JsonResponse(
            {"error": "Debe proporcionar un archivo PDF"}, status=400
        )
        await response(scope, receive, send)
        return

    pdf_file = request.FILES["pdf"]
    start_time = time.time()

    try:
        temp_path = default_storage.save("temp/temp.pdf", ContentFile(pdf_file.read()))
        temp_file_path = default_storage.path(temp_path)

        # Usar ThreadPoolExecutor para procesar el PDF de manera asíncrona
        problemas_detectados = await asyncio.to_thread(
            Classifier.process_pdf, temp_file_path
        )

        # Guardar problemas en la base de datos
        for problema in problemas_detectados:
            await asyncio.to_thread(
                collection.update_one,
                {"formula": problema["formula"]},
                {"$setOnInsert": problema},
                upsert=True,
            )

        end_time = time.time()
        logger.info(
            f"Tiempo de procesamiento del PDF: {end_time - start_time:.2f} segundos"
        )
        response = JsonResponse(
            {
                "message": f"PDF '{pdf_file.name}' procesado correctamente",
                "problemas": problemas_detectados,
            }
        )
        await response(scope, receive, send)
    except Exception as e:
        logger.error(f"Error al procesar el PDF: {str(e)}")
        response = JsonResponse(
            {"error": f"Error al procesar el PDF: {str(e)}"}, status=500
        )
        await response(scope, receive, send)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def procesar_imagen(scope, receive, send):
    """Procesa una imagen subida y detecta ecuaciones matemáticas."""
    request = ASGIRequest(scope, receive)

    if "imagen" not in request.FILES:
        response = JsonResponse(
            {"error": "Debe proporcionar un archivo de imagen"}, status=400
        )
        await response(scope, receive, send)
        return

    imagen = request.FILES["imagen"]
    start_time = time.time()

    try:
        temp_path = default_storage.save(
            "temp/temp_image.png", ContentFile(imagen.read())
        )
        temp_file_path = default_storage.path(temp_path)

        # Usar caché para evitar reprocesamiento de imágenes idénticas con redis
        file_hash = hashlib.md5(imagen.read()).hexdigest()
        cached_result = cache.get(file_hash)
        if cached_result:
            logger.info(f"Resultado obtenido de caché para {file_hash}")
            response = JsonResponse(cached_result)
            await response(scope, receive, send)
            return

        cleaned_formula, problem_type = await asyncio.to_thread(
            Classifier.process_image, temp_file_path
        )
        if cleaned_formula:
            # Renderizar la fórmula LaTeX a imagen
            latex_image = await asyncio.to_thread(
                render_latex_to_image, cleaned_formula
            )
            buffered = io.BytesIO()
            latex_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Guardar problema en la base de datos
            await asyncio.to_thread(
                collection.update_one,
                {"formula": cleaned_formula},
                {
                    "$setOnInsert": {
                        "imagen": imagen.name,
                        "formula": cleaned_formula,
                        "tipo": problem_type,
                    }
                },
                upsert=True,
            )

            response_data = {
                "message": f"Imagen '{imagen.name}' procesada correctamente",
                "problema": {
                    "formula": cleaned_formula,
                    "tipo": problem_type,
                    "latex_image": img_str,
                },
            }

            # Guardar en caché el resultado
            cache.set(file_hash, response_data, timeout=3600)  # Caché por 1 hora

            end_time = time.time()
            logger.info(
                f"Tiempo de procesamiento de la imagen: {end_time - start_time:.2f} segundos"
            )
            response = JsonResponse(response_data)
            await response(scope, receive, send)
            return

        logger.warning("No se detectó ninguna fórmula en la imagen.")
        response = JsonResponse(
            {"error": "No se detectó ninguna fórmula en la imagen."}, status=404
        )
        await response(scope, receive, send)

    except Exception as e:
        logger.error(f"Error al procesar la imagen: {str(e)}")
        response = JsonResponse(
            {"error": f"Error al procesar la imagen: {str(e)}"}, status=500
        )
        await response(scope, receive, send)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@csrf_exempt  # Para desactivar CSRF en esta vista (usualmente en desarrollo)
def procesar_texto(request):
    # Verificar que la solicitud es POST
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        # Cargar el contenido del cuerpo de la solicitud JSON
        data = json.loads(request.body)
        texto = data.get("texto", "")  # Obtener el texto del cuerpo de la solicitud

        # Convertir el texto a LaTeX
        latex = convertir_a_latex(texto)

        # Responder con el resultado en formato JSON
        return JsonResponse({"latex": latex}, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON inválido"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# Función para manejar el cierre del ThreadPoolExecutor
def cleanup():
    executor.shutdown(wait=True)


# Registrar la función de limpieza para que se ejecute al cerrar la aplicación
import atexit

atexit.register(cleanup)
