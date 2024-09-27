from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import os
import logging
import cv2
from PIL import Image
from PyPDF2 import PdfReader
from .model import Classifier, preprocess_image
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient , errors
import re


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
def abrir_camara(request):
    """Abre la cámara, procesa imágenes y detecta problemas matemáticos en tiempo real."""
    if request.method == "POST":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        detected_problems = []
        process_interval = 1  # Procesar cada 5 cuadros
        frame_count = 0

        if not cap.isOpened():
            return JsonResponse({"error": "Error al abrir la cámara"}, status=500)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Procesar cada N cuadros
                if frame_count % process_interval == 0:
                    # Preprocesar el frame
                    preprocessed_frame = preprocess_image(frame)

                    # Guardar temporalmente la imagen procesada
                    temp_image_path = "temp/temp_frame.png"
                    cv2.imwrite(temp_image_path, preprocessed_frame)

                    # Clasificar problema usando el modelo
                    cleaned_formula, problem_type = Classifier.classify_problem(
                        temp_image_path
                    )

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
                            detected_problems.append(
                                {"formula": cleaned_formula, "tipo": problem_type}
                            )
                        else:
                            logging.info(
                                f"Problema existente actualizado: {cleaned_formula}"
                            )

                # Dibujar un recuadro y mostrar el texto en la imagen
                for idx, problem in enumerate(detected_problems):
                    cv2.rectangle(
                        frame, (50, 50 + idx * 30), (600, 80 + idx * 30), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{problem['formula']} ({problem['tipo']})",
                        (60, 70 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Mostrar la imagen en tiempo real
                cv2.imshow("Cámara en tiempo real", frame)

                # Romper el bucle si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return JsonResponse(
            {
                "message": "Cámara cerrada correctamente",
                "problemas_detectados": detected_problems,
            }
        )
    return JsonResponse({"error": "Método no permitido"}, status=405)


@csrf_exempt
def procesar_pdf(request):
    """Procesa un archivo PDF subido y extrae ecuaciones matemáticas."""
    if request.method == "POST" and "pdf" in request.FILES:
        pdf_file = request.FILES["pdf"]
        try:
            pdf_reader = PdfReader(pdf_file)
            problemas_detectados = []

            for i, page in enumerate(pdf_reader.pages):
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
            return JsonResponse({"error": "No se detectó ninguna fórmula"}, status=404)

        except Exception as e:
            logging.error(f"Error al procesar la imagen: {str(e)}")
            return JsonResponse(
                {"error": f"Error al procesar la imagen: {str(e)}"}, status=500
            )
    return JsonResponse({"error": "Debe proporcionar un archivo de imagen"}, status=400)
