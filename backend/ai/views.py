from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.cache import cache
from django_ratelimit.decorators import ratelimit
from pymongo import MongoClient, errors
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from .model import Classifier, preprocess_image, render_latex_to_image
import os
import logging
import cv2
import base64
import io
import time
import asyncio

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración de MongoDB
mongo_uri = os.getenv("MONGO_URI")
try:
    client = MongoClient(mongo_uri)
    db = client["math_problems_db"]
    collection = db["problemas"]
    logger.info(f"Conexión exitosa a MongoDB en {mongo_uri}")
except errors.ConnectionFailure as e:
    logger.error(f"No se pudo conectar a MongoDB: {e}")
    raise

# Crear un ThreadPoolExecutor global
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

def index(request):
    """Renderiza la página principal con las opciones disponibles."""
    return render(request, "index.html")

@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
@ratelimit(key='ip', rate='10/m', block=True)
async def procesar_fotograma(request):
    """Recibe un fotograma enviado por el frontend y detecta problemas matemáticos."""
    if request.method == "OPTIONS":
        response = JsonResponse({'message': 'OK'})
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    if "frame" not in request.FILES:
        return JsonResponse({"error": "Debe proporcionar un fotograma en la solicitud."}, status=400)

    frame_file = request.FILES["frame"]
    start_time = time.time()
    
    try:
        # Guardar el archivo temporalmente
        temp_path = default_storage.save('temp/temp_frame.png', ContentFile(frame_file.read()))
        temp_file_path = default_storage.path(temp_path)

        # Usar caché para evitar reprocesamiento de imágenes idénticas
        file_hash = hash(frame_file.read())
        cached_result = cache.get(file_hash)
        if cached_result:
            logger.info(f"Resultado obtenido de caché para {file_hash}")
            return JsonResponse(cached_result)

        # Preprocesar la imagen
        img = cv2.imread(temp_file_path)
        preprocessed_img = await asyncio.to_thread(preprocess_image, img)
        cv2.imwrite(temp_file_path, preprocessed_img)

        # Clasificar problema usando el modelo
        cleaned_formula, problem_type = await asyncio.to_thread(Classifier.classify_problem, temp_file_path)

        if cleaned_formula:
            # Renderizar la fórmula LaTeX a imagen
            latex_image = await asyncio.to_thread(render_latex_to_image, cleaned_formula)
            buffered = io.BytesIO()
            latex_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Intentar agregar el problema a la base de datos usando upsert
            result = await asyncio.to_thread(
                collection.update_one,
                {"formula": cleaned_formula},
                {
                    "$setOnInsert": {
                        "formula": cleaned_formula,
                        "tipo": problem_type,
                        "usos": 0,
                    },
                },
                upsert=True
            )

            message = "Nuevo problema detectado y agregado a la base de datos." if result.upserted_id else "Problema ya existente en la base de datos."
            
            response_data = {
                "formula": cleaned_formula,
                "tipo": problem_type,
                "message": message,
                "latex_image": img_str
            }

            # Guardar en caché el resultado
            cache.set(file_hash, response_data, timeout=3600)  # Caché por 1 hora

            end_time = time.time()
            logger.info(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")
            return JsonResponse(response_data)
        
        logger.warning("No se detectó ninguna fórmula en el fotograma.")
        return JsonResponse({"error": "No se detectó ninguna fórmula en el fotograma."}, status=404)

    except Exception as e:
        logger.error(f"Error al procesar el fotograma: {str(e)}")
        return JsonResponse({"error": f"Error al procesar el fotograma: {str(e)}"}, status=500)
    finally:
        # Limpiar archivos temporales
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key='ip', rate='5/m', block=True)
async def procesar_pdf(request):
    """Procesa un archivo PDF subido y extrae ecuaciones matemáticas."""
    if "pdf" not in request.FILES:
        return JsonResponse({"error": "Debe proporcionar un archivo PDF"}, status=400)

    pdf_file = request.FILES["pdf"]
    start_time = time.time()
    
    try:
        temp_path = default_storage.save('temp/temp.pdf', ContentFile(pdf_file.read()))
        temp_file_path = default_storage.path(temp_path)
        
        # Usar ThreadPoolExecutor para procesar el PDF de manera asíncrona
        problemas_detectados = await asyncio.to_thread(Classifier.process_pdf, temp_file_path)

        # Guardar problemas en la base de datos
        for problema in problemas_detectados:
            await asyncio.to_thread(
                collection.update_one,
                {"formula": problema["formula"]},
                {"$setOnInsert": problema},
                upsert=True
            )

        end_time = time.time()
        logger.info(f"Tiempo de procesamiento del PDF: {end_time - start_time:.2f} segundos")
        return JsonResponse({
            "message": f"PDF '{pdf_file.name}' procesado correctamente",
            "problemas": problemas_detectados,
        })
    except Exception as e:
        logger.error(f"Error al procesar el PDF: {str(e)}")
        return JsonResponse({"error": f"Error al procesar el PDF: {str(e)}"}, status=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key='ip', rate='10/m', block=True)
async def procesar_imagen(request):
    """Procesa una imagen subida y detecta ecuaciones matemáticas."""
    if "imagen" not in request.FILES:
        return JsonResponse({"error": "Debe proporcionar un archivo de imagen"}, status=400)

    imagen = request.FILES["imagen"]
    start_time = time.time()
    
    try:
        temp_path = default_storage.save('temp/temp_image.png', ContentFile(imagen.read()))
        temp_file_path = default_storage.path(temp_path)

        # Usar caché para evitar reprocesamiento de imágenes idénticas
        file_hash = hash(imagen.read())
        cached_result = cache.get(file_hash)
        if cached_result:
            logger.info(f"Resultado obtenido de caché para {file_hash}")
            return JsonResponse(cached_result)

        cleaned_formula, problem_type = await asyncio.to_thread(Classifier.process_image, temp_file_path)
        if cleaned_formula:
            # Renderizar la fórmula LaTeX a imagen
            latex_image = await asyncio.to_thread(render_latex_to_image, cleaned_formula)
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
                upsert=True
            )

            response_data = {
                "message": f"Imagen '{imagen.name}' procesada correctamente",
                "problema": {
                    "formula": cleaned_formula,
                    "tipo": problem_type,
                    "latex_image": img_str
                },
            }

            # Guardar en caché el resultado
            cache.set(file_hash, response_data, timeout=3600)  # Caché por 1 hora

            end_time = time.time()
            logger.info(f"Tiempo de procesamiento de la imagen: {end_time - start_time:.2f} segundos")
            return JsonResponse(response_data)
        
        logger.warning("No se detectó ninguna fórmula en la imagen.")
        return JsonResponse({"error": "No se detectó ninguna fórmula en la imagen."}, status=404)

    except Exception as e:
        logger.error(f"Error al procesar la imagen: {str(e)}")
        return JsonResponse({"error": f"Error al procesar la imagen: {str(e)}"}, status=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Función para manejar el cierre del ThreadPoolExecutor
def cleanup():
    executor.shutdown(wait=True)

# Registrar la función de limpieza para que se ejecute al cerrar la aplicación
import atexit
atexit.register(cleanup)
