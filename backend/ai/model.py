"""
Módulo de inteligencia artificial para el reconocimiento y clasificación de ecuaciones matemáticas.

Este módulo se encarga de cargar y entrenar el modelo utilizando transformers.
"""

import logging
import os
import cv2
import re
import io
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configuración del logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir la resolución de captura
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0

# Crear carpeta temporal para guardar frames
if not os.path.exists("temp"):
    os.mkdir("temp")

# Establecer variable de entorno para evitar warnings de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ruta de la carpeta para los modelos
model_dir = os.path.join("ai", "Models")
# Crear la carpeta si no existe
os.makedirs(model_dir, exist_ok=True)

# Cargando el tokenizador y el modelo para visión
logging.info("Iniciando la carga del modelo de visión.")
try:
    model = VisionEncoderDecoderModel.from_pretrained(
        "DGurgurov/im2latex", cache_dir=model_dir, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex", cache_dir=model_dir)
    feature_extractor = ViTImageProcessor.from_pretrained(
        "microsoft/swin-base-patch4-window7-224-in22k", cache_dir=model_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Modelo de visión cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el modelo de visión: {e}")
    raise e


def preprocess_image(image):
    """
    Procesa la imagen para mejorar la detección de fórmulas LaTeX.
    Convierte a escala de grises, aplica desenfoque y ajuste de contraste.
    """
    try:
        logging.info("Preprocesando la imagen...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, (3, 3), 0
        )  # Reduce el kernel para mantener detalles
        _, binary = cv2.threshold(
            blurred, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Kernel más pequeño
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contrast = cv2.convertScaleAbs(
            denoised, alpha=1.2, beta=10
        )  # Ajustar contraste
        logging.info("Preprocesamiento de imagen completado.")
        return contrast
    except Exception as e:
        logging.error(f"Error en el preprocesamiento de la imagen: {e}")
        return None


def process_with_im2latex(image_path):
    """
    Procesa la imagen usando el modelo Im2Latex y devuelve la fórmula detectada.
    """
    try:
        logging.info(f"Procesando imagen con el modelo Im2Latex: {image_path}")
        image = Image.open(image_path).convert("RGB")
        pixel_values = feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device)
        # Generar máscara de atención para la entrada
        attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long).to(device)
        generated_ids = model.generate(pixel_values, attention_mask=attention_mask)

        # Decodificar la secuencia generada
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        logging.info(f"Fórmula LaTeX detectada: {generated_text}")
        return generated_text
    except Exception as e:
        logging.error(f"Error al procesar la imagen con Im2Latex: {e}")
        return None


def correct_ocr_errors(text):
    """
    Corrige errores comunes de OCR en las fórmulas detectadas.
    """
    logging.info("Corrigiendo errores comunes de OCR en la fórmula detectada.")
    corrections = {"—": "-", "vb": "√", " ": ""}
    for wrong_char, correct_char in corrections.items():
        text = text.replace(wrong_char, correct_char)
    return text


def clean_latex_text(latex_text):
    """
    Limpia el texto LaTeX eliminando espacios adicionales.
    """
    logging.info("Limpiando el texto LaTeX.")
    latex_text = re.sub(r"\s+", " ", latex_text).strip()
    return latex_text


def render_latex_to_image(latex_text):
    """
    Renderiza la fórmula LaTeX a una imagen utilizando matplotlib.
    """
    logging.info(f"Renderizando la fórmula LaTeX a imagen: {latex_text}")
    try:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"${latex_text}$", fontsize=20, ha="center", va="center")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        buf.close()
        logging.info("Renderización completada.")
        return img
    except Exception as e:
        logging.error(f"Error al renderizar LaTeX a imagen: {e}")
        return None
      



def classify_problem_type(latex_formula):
    """
    Clasifica el tipo de problema matemático basado en la fórmula LaTeX.
    """
    if re.search(r"\\int", latex_formula):
        return "Integral"
    elif re.search(r"\\lim", latex_formula):
        return "Límite"
    elif re.search(r"\\frac{d}{dx}", latex_formula):
        return "Derivada"
    elif re.search(r"\\sum", latex_formula):
        return "Suma"
    elif re.search(r"\\prod", latex_formula):
        return "Producto"
    elif re.search(r"=", latex_formula):
        return "Ecuación"
    else:
        return "Desconocido"




class Classifier:
    @staticmethod
    def classify_problem(image_path):
        latex_formula = process_with_im2latex(image_path)
        if latex_formula:
            cleaned_formula = clean_latex_text(latex_formula)
            problem_type = classify_problem_type(cleaned_formula)
            return cleaned_formula, problem_type
        return "", "Desconocido"


def main():
    logging.info("Módulo de inteligencia artificial iniciado.")
    # Aquí puedes agregar cualquier lógica adicional que quieras ejecutar al iniciar el módulo


if __name__ == "__main__":
    main()
