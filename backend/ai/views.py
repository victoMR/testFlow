from django.shortcuts import render
from django.http import JsonResponse
import os
import logging
import cv2
from PIL import Image
from PyPDF2 import PdfReader
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient, errors
import re
from .model import Classifier

# Cargar variables de entorno desde el archivo .env
from dotenv import load_dotenv

load_dotenv()

# Configurar la conexión a MongoDB
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(os.getenv("MONGO_URI"))
db = client["math_problems_db"]
collection = db["problemas"]

# Verificación de conexión a MongoDB
try:
    client.server_info()  # Esto arroja una excepción si no se puede conectar
    logging.info(f"Conexión exitosa a MongoDB en {mongo_uri}")
except errors.ServerSelectionTimeoutError as e:
    logging.error(f"No se pudo conectar a MongoDB: {e}")
except errors.InvalidURI as e:
    logging.error(f"URI de MongoDB inválida: {e}")
except Exception as e:
    logging.error(f"Error desconocido al conectar a MongoDB: {e}")

# Configuración del logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def index(request):
    """Renderiza la página principal con las opciones disponibles."""
    return render(request, "index.html")


@csrf_exempt
def procesar_fotograma(request):
    """Recibe un fotograma enviado por el frontend y detecta problemas matemáticos."""
    if request.method == "POST" and "frame" in request.FILES:
        frame_file = request.FILES["frame"]
        try:
            # Leer la imagen recibida y guardarla temporalmente
            frame = Image.open(frame_file)
            temp_image_path = "temp/temp_frame.png"
            frame.save(temp_image_path)

            # Clasificar problema usando el modelo
            cleaned_formula, problem_type = Classifier.classify_problem(temp_image_path)

            if cleaned_formula:
                # Intentar agregar el problema a la base de datos usando upsert
                result = collection.update_one(
                    {"formula": cleaned_formula},  # Buscar por fórmula
                    {
                        "$setOnInsert": {  # Solo se agrega si no existe
                            "formula": cleaned_formula,
                            "tipo": problem_type,
                            "usos": 0,  # Inicialmente, los usos son 0
                        },
                    },
                    upsert=True,  # Insertar si no existe
                )

                # Verificar si se insertó un nuevo documento o se actualizó uno existente
                if (
                    result.upserted_id
                ):  # Si se insertó uno nuevo, `upserted_id` tendrá valor
                    logging.info(
                        f"Nuevo problema agregado a la base de datos: {cleaned_formula}"
                    )
                    return JsonResponse(
                        {
                            "formula": cleaned_formula,
                            "tipo": problem_type,
                            "message": "Nuevo problema detectado y agregado a la base de datos.",
                        }
                    )
                else:
                    logging.info(f"Problema existente: {cleaned_formula}")
                    return JsonResponse(
                        {
                            "formula": cleaned_formula,
                            "tipo": problem_type,
                            "message": "Problema ya existente en la base de datos.",
                        }
                    )
            return JsonResponse(
                {"error": "No se detectó ninguna fórmula en el fotograma."}, status=404
            )

        except Exception as e:
            logging.error(f"Error al procesar el fotograma: {str(e)}")
            return JsonResponse(
                {"error": f"Error al procesar el fotograma: {str(e)}"}, status=500
            )

    return JsonResponse(
        {"error": "Debe proporcionar un fotograma en la solicitud."}, status=400
    )


@csrf_exempt
def procesar_pdf(request):
    """Procesa un archivo PDF subido y extrae ecuaciones matemáticas."""
    if request.method == "POST" and "pdf" in request.FILES:
        pdf_file = request.FILES["pdf"]
        try:
            pdf_reader = PdfReader(pdf_file)
            problemas_detectados = []

            for page in pdf_reader.pages:
                text = page.extract_text() if page.extract_text() else ""
                # Extracción de fórmulas del texto usando expresiones regulares
                formulas = re.findall(r"[0-9\+\-\*\/\=\(\)\s]+", text)
                for formula in formulas:
                    cleaned_formula = Classifier.clean_latex_text(formula)
                    problem_type = Classifier.classify_problem_type(cleaned_formula)
                    problemas_detectados.append(
                        {"formula": cleaned_formula, "tipo": problem_type}
                    )

            # Guardar problemas en la base de datos
            collection.insert_many(problemas_detectados)

            return JsonResponse(
                {
                    "message": f"PDF '{pdf_file.name}' procesado correctamente",
                    "problemas": problemas_detectados,
                }
            )
        except Exception as e:
            logging.error(f"Error al procesar el PDF: {str(e)}")
            return JsonResponse(
                {"error": f"Error al procesar el PDF: {str(e)}"}, status=500
            )
    return JsonResponse({"error": "Debe proporcionar un archivo PDF"}, status=400)


@csrf_exempt
def procesar_imagen(request):
    """Procesa una imagen subida y detecta ecuaciones matemáticas."""
    if request.method == "POST" and "imagen" in request.FILES:
        imagen = request.FILES["imagen"]
        try:
            img = Image.open(imagen)
            # Guardar la imagen temporalmente
            temp_image_path = "temp/temp_image.png"
            img.save(temp_image_path)

            cleaned_formula, problem_type = Classifier.classify_problem(temp_image_path)
            if cleaned_formula:
                # Guardar problema en la base de datos
                collection.insert_one(
                    {
                        "imagen": imagen.name,
                        "formula": cleaned_formula,
                        "tipo": problem_type,
                    }
                )

                return JsonResponse(
                    {
                        "message": f"Imagen '{imagen.name}' procesada correctamente",
                        "problema": {"formula": cleaned_formula, "tipo": problem_type},
                    }
                )
            return JsonResponse(
                {"error": "No se detectó ninguna fórmula en la imagen."}, status=404
            )

        except Exception as e:
            logging.error(f"Error al procesar la imagen: {str(e)}")
            return JsonResponse(
                {"error": f"Error al procesar la imagen: {str(e)}"}, status=500
            )
    return JsonResponse({"error": "Debe proporcionar un archivo de imagen"}, status=400)
