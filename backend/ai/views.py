from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from .model import (
    FormulaDetector,
    formula_detector,
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
    client = MongoClient(
        mongo_uri,
        tls=True,  # Usar tls en lugar de ssl
        tlsAllowInvalidCertificates=True,  # Permitir certificados no válidos
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        retryWrites=True,
        w='majority'
    )
    # Verificar la conexión
    client.admin.command('ping')
    
    db = client["math_problems_db"]
    collection = db["problemas"]
    logger.info("Conexión exitosa a MongoDB")
except errors.ConnectionFailure as e:
    logger.error(f"No se pudo conectar a MongoDB: {e}")
    # En lugar de levantar una excepción, crear una colección en memoria
    class MemoryCollection:
        def __init__(self):
            self.documents = []
            
        def insert_one(self, document):
            self.documents.append(document)
            return True
            
        def find(self):
            return self.documents
    
    logger.warning("Usando almacenamiento en memoria como fallback")
    collection = MemoryCollection()

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
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-methods", b"POST, OPTIONS"),
                (b"access-control-allow-headers", b"content-type"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": b"",
        })
        return

    if scope["method"] != "POST":
        await send({
            "type": "http.response.start",
            "status": 405,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({"type": "http.response.body", "body": b"Method Not Allowed"})
        return

    try:
        # Leer el cuerpo de la solicitud
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        # Procesar la imagen
        nparr = np.frombuffer(body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen")

        # Solo detectar la fórmula
        latex_formula = await asyncio.to_thread(
            formula_detector.process_image, img
        )

        if latex_formula:
            # Guardar en la base de datos
            await asyncio.to_thread(
                collection.insert_one,
                {
                    "formula": latex_formula,
                    "fecha_deteccion": time.time(),
                    "procesado": False
                }
            )

            response_data = {
                "status": "success",
                "message": "Fórmula detectada y guardada"
            }

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"access-control-allow-origin", b"http://localhost:3000"),
                    (b"access-control-allow-credentials", b"true"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps(response_data).encode(),
            })
            return

        # Respuesta cuando no se detecta fórmula
        response_data = {
            "status": "no_formula",
            "message": "No se detectó ninguna fórmula matemática en la imagen",
            "details": {
                "suggestion": "Asegúrate de mostrar claramente una fórmula matemática a la cámara"
            }
        }

        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(response_data).encode(),
        })

    except Exception as e:
        logger.error(f"Error al procesar el fotograma: {e}")
        error_response = {
            "status": "error",
            "message": "Error al procesar la imagen",
            "details": {
                "error": str(e),
                "suggestion": "Por favor, intenta de nuevo o verifica la conexión"
            }
        }
        
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(error_response).encode(),
        })


async def procesar_pdf(scope, receive, send):
    """Procesa un archivo PDF subido y extrae ecuaciones matemáticas."""
    request = ASGIRequest(scope, receive)

    if scope["method"] == "OPTIONS":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-methods", b"POST, OPTIONS"),
                (b"access-control-allow-headers", b"content-type"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": b"",
        })
        return

    if "pdf" not in request.FILES:
        response_data = {
            "status": "error",
            "message": "Debe proporcionar un archivo PDF"
        }
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(response_data).encode(),
        })
        return

    pdf_file = request.FILES["pdf"]
    start_time = time.time()

    try:
        # Guardar temporalmente el archivo PDF
        temp_path = default_storage.save("temp/temp.pdf", ContentFile(pdf_file.read()))
        temp_file_path = default_storage.path(temp_path)

        # Usar la instancia global formula_detector para procesar el PDF
        problemas_detectados = await asyncio.to_thread(
            formula_detector.process_pdf, temp_file_path
        )

        # Guardar problemas en la base de datos
        for problema in problemas_detectados:
            await asyncio.to_thread(
                collection.insert_one,
                {
                    "formula": problema["formula"],
                    "tipo": problema["tipo"],
                    "pagina": problema["pagina"],
                    "confidence": problema["confidence"],
                    "origen": problema["origen"],
                    "fecha_deteccion": time.time(),
                    "nombre_archivo": pdf_file.name
                }
            )

        end_time = time.time()
        tiempo_proceso = end_time - start_time
        
        response_data = {
            "status": "success",
            "message": f"PDF '{pdf_file.name}' procesado correctamente",
            "tiempo_proceso": f"{tiempo_proceso:.2f} segundos",
            "problemas": problemas_detectados,
            "total_formulas": len(problemas_detectados)
        }

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(response_data).encode(),
        })

    except Exception as e:
        logger.error(f"Error al procesar el PDF: {str(e)}")
        error_response = {
            "status": "error",
            "message": "Error al procesar el PDF",
            "details": {
                "error": str(e),
                "suggestion": "Verifica que el archivo sea un PDF válido y no esté dañado"
            }
        }
        
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(error_response).encode(),
        })

    finally:
        # Limpiar archivos temporales
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def procesar_imagen(scope, receive, send):
    """Procesa una imagen subida y detecta ecuaciones matemáticas."""
    
    if scope["method"] == "OPTIONS":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-methods", b"POST, OPTIONS"),
                (b"access-control-allow-headers", b"content-type"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": b"",
        })
        return

    try:
        # Leer el cuerpo de la solicitud
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        # Buscar el inicio y fin de los datos de la imagen
        start_marker = b'\r\n\r\n'
        end_marker = b'\r\n--'
        
        start_idx = body.find(start_marker)
        if start_idx == -1:
            raise ValueError("No se encontró el inicio de los datos de la imagen")
        
        start_idx += len(start_marker)
        
        end_idx = body.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = len(body)  # Si no hay marcador final, usar todo el contenido restante

        # Extraer los datos binarios de la imagen
        image_data = body[start_idx:end_idx]
        
        if not image_data:
            raise ValueError("No se encontraron datos de imagen")

        # Verificar tamaño antes de procesar
        if len(image_data) > 5 * 1024 * 1024:  # 5MB
            raise ValueError("La imagen no debe superar 5MB")

        # Convertir a numpy array y decodificar
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen. Formato no soportado o datos corruptos.")

        start_time = time.time()
        logger.info("Imagen decodificada correctamente. Iniciando procesamiento...")

        # Detectar fórmula
        latex_formula = await asyncio.to_thread(
            formula_detector.process_image, img
        )

        if latex_formula:
            # Clasificar tipo de problema
            problem_type = formula_detector.latex_processor.classify_problem(latex_formula)
            
            # Calcular confianza
            confidence = await asyncio.to_thread(
                formula_detector._calculate_confidence, latex_formula
            )

            try:
                # Intentar guardar en MongoDB con retry
                for attempt in range(3):  # Intentar 3 veces
                    try:
                        await asyncio.to_thread(
                            collection.insert_one,
                            {
                                "formula": latex_formula,
                                "tipo": problem_type,
                                "confidence": confidence,
                                "fecha_deteccion": time.time(),
                                "procesado": True
                            }
                        )
                        break  # Si tiene éxito, salir del bucle
                    except Exception as db_error:
                        if attempt == 2:  # Si es el último intento
                            logger.error(f"Error al guardar en la base de datos: {db_error}")
                        else:
                            await asyncio.sleep(1)  # Esperar antes de reintentar
                            continue

            except Exception as db_error:
                logger.error(f"Error al guardar en la base de datos: {db_error}")
                # Continuar con el procesamiento aunque falle el guardado

            # Renderizar fórmula y continuar con la respuesta
            latex_image = await asyncio.to_thread(
                formula_detector.render_latex, latex_formula
            )
            
            if latex_image:
                buffered = io.BytesIO()
                latex_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
            else:
                img_str = None

            end_time = time.time()
            response_data = {
                "status": "success",
                "message": "Imagen procesada correctamente",
                "tiempo_proceso": f"{end_time - start_time:.2f} segundos",
                "problema": {
                    "formula": latex_formula,
                    "tipo": problem_type,
                    "confidence": confidence,
                    "latex_image": img_str
                }
            }

            logger.info(f"Procesamiento exitoso. Fórmula detectada: {latex_formula}")

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"access-control-allow-origin", b"http://localhost:3000"),
                    (b"access-control-allow-credentials", b"true"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps(response_data).encode(),
            })
            return

        # No se detectó fórmula
        response_data = {
            "status": "no_formula",
            "message": "No se detectó ninguna fórmula matemática en la imagen",
            "details": {
                "suggestion": "Asegúrate de que la imagen contenga una fórmula matemática clara"
            }
        }

        logger.info("No se detectó fórmula matemática en la imagen")

        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(response_data).encode(),
        })

    except Exception as e:
        logger.error(f"Error al procesar la imagen: {str(e)}")
        error_response = {
            "status": "error",
            "message": "Error al procesar la imagen",
            "details": {
                "error": str(e),
                "suggestion": "Verifica el formato y tamaño de la imagen"
            }
        }
        
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"http://localhost:3000"),
                (b"access-control-allow-credentials", b"true"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps(error_response).encode(),
        })


async def procesar_texto(scope, receive, send):
    if scope['method'] == 'OPTIONS':
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [
                (b'content-type', b'application/json'),
                (b'access-control-allow-origin', b'http://localhost:3000'),
                (b'access-control-allow-methods', b'POST, OPTIONS'),
                (b'access-control-allow-headers', b'Content-Type'),
                (b'access-control-allow-credentials', b'true'),
            ],
        })
        await send({
            'type': 'http.response.body',
            'body': b'',
        })
        return

    if scope['method'] != 'POST':
        await send({
            'type': 'http.response.start',
            'status': 405,
            'headers': [(b'content-type', b'application/json')],
        })
        await send({
            'type': 'http.response.body',
            'body': json.dumps({"error": "Método no permitido"}).encode(),
        })
        return

    body = b''
    more_body = True
    while more_body:
        message = await receive()
        body += message.get('body', b'')
        more_body = message.get('more_body', False)

    try:
        data = json.loads(body)
        texto = data.get("texto", "")

        logger.info(f"Texto recibido: '{texto}'")

        # Convertir el texto a LaTeX
        latex = formula_detector.convertir_a_latex(texto)

        logger.info(f"LaTeX generado: '{latex}'")

        # Responder con el resultado en formato JSON
        response_data = json.dumps({"latex": latex})
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [
                (b'content-type', b'application/json'),
                (b'access-control-allow-origin', b'http://localhost:3000'),
                (b'access-control-allow-credentials', b'true'),
            ],
        })
        await send({
            'type': 'http.response.body',
            'body': response_data.encode(),
        })

    except json.JSONDecodeError:
        logger.error("Error: JSON inválido")
        await send({
            'type': 'http.response.start',
            'status': 400,
            'headers': [(b'content-type', b'application/json')],
        })
        await send({
            'type': 'http.response.body',
            'body': json.dumps({"error": "JSON inválido"}).encode(),
        })

    except Exception as e:
        logger.error(f"Error al procesar el texto: {str(e)}")
        await send({
            'type': 'http.response.start',
            'status': 500,
            'headers': [(b'content-type', b'application/json')],
        })
        await send({
            'type': 'http.response.body',
            'body': json.dumps({"error": str(e)}).encode(),
        })


# Función para manejar el cierre del ThreadPoolExecutor
def cleanup():
    executor.shutdown(wait=True)


# Registrar la función de limpieza para que se ejecute al cerrar la aplicación
import atexit

atexit.register(cleanup)























