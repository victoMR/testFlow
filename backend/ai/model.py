"""
Módulo de inteligencia artificial para el reconocimiento y clasificación de ecuaciones matemáticas.

Este módulo se encarga de cargar y entrenar el modelo utilizando transformers, 
implementa caché con Redis, procesamiento en hilos, y logging avanzado.
"""

import logging
import os
import cv2
import re
import io
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sympy import sympify, latex
from sympy.parsing.latex import parse_latex
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from redis import Redis
from redis.exceptions import (
    ConnectionError,
    TimeoutError as RedisTimeoutError,
    AuthenticationError,
)
import json
from functools import lru_cache
import requests
import hashlib
from io import BytesIO
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_model.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Configuración de Redis con reintentos
def get_redis_connection():
    for _ in range(3):  # Intentar 3 veces
        try:
            redis_client = Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                db=int(os.getenv("REDIS_DB", 0)),
                password=os.getenv("REDIS_PASSWORD"),
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            redis_client.ping()  # Verificar la conexión
            logger.info(
                f"Conectado a Redis en {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')} con base de datos {os.getenv('REDIS_DB')}"
            )
            return redis_client
        except (ConnectionError, RedisTimeoutError, AuthenticationError) as e:
            logger.warning(f"Intento de conexión a Redis fallido: {e}")

    logger.warning("No se pudo conectar a Redis. La aplicación funcionará sin caché.")
    return None


# Usa esta variable global para verificar si Redis está disponible
redis_available = get_redis_connection() is not None

# Configuración del modelo
model_dir = os.path.join("ai", "Models")
os.makedirs(model_dir, exist_ok=True)

# ThreadPoolExecutor para procesamiento paralelo
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


# Carga del modelo (con caché de Redis)
@lru_cache(maxsize=1)
def load_model():
    logger.info("Iniciando la carga del modelo de visión.")
    try:
        # Cargar el modelo normalmente, sin usar caché de Redis para el modelo completo
        model = VisionEncoderDecoderModel.from_pretrained(
            "DGurgurov/im2latex", cache_dir=model_dir, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "DGurgurov/im2latex", cache_dir=model_dir
        )
        feature_extractor = ViTImageProcessor.from_pretrained(
            "microsoft/swin-base-patch4-window7-224-in22k", cache_dir=model_dir
        )

        logger.info("Modelo de visión cargado exitosamente.")
        return model, tokenizer, feature_extractor
    except Exception as e:
        logger.error(f"Error al cargar el modelo de visión: {e}")
        raise


model, tokenizer, feature_extractor = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_image(image):
    """
    Procesa la imagen para mejorar la detección de fórmulas LaTeX.
    Optimizado para fórmulas matem��ticas y eliminación de elementos no deseados.
    """
    try:
        logger.info("Preprocesando la imagen")
        if isinstance(image, np.ndarray):
            # Convertir a escala de grises si es necesario
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Aplicar umbral adaptativo para mejorar el contraste de texto
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Eliminar ruido manteniendo texto
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

            # Detectar y eliminar áreas grandes que probablemente no sean fórmulas
            contours, _ = cv2.findContours(
                255 - denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filtrar contornos basados en área y proporción
            mask = np.ones_like(denoised) * 255
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                # Si el área es muy grande o la proporción no es típica de una fórmula
                if area > denoised.size * 0.5 or aspect_ratio > 5 or aspect_ratio < 0.2:
                    cv2.drawContours(mask, [cnt], -1, 0, -1)

            # Aplicar máscara
            result = cv2.bitwise_and(denoised, mask)

            # Mejorar contraste final
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            final = clahe.apply(result)

            logger.info("Preprocesamiento de imagen completado exitosamente")
            return final
        else:
            logger.error("La imagen no es un array de numpy válido")
            return None

    except Exception as e:
        logger.error(f"Error en el preprocesamiento de la imagen: {e}")
        return None


def process_with_im2latex(image):
    """
    Procesa la imagen para detectar y convertir fórmulas matemáticas a LaTeX.
    Optimizado para velocidad y precisión.
    """
    try:
        logger.info("Procesando imagen con el modelo Im2Latex")

        # Preprocesar imagen
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            raise ValueError("Error en el preprocesamiento de la imagen")

        # Convertir a formato PIL
        image = Image.fromarray(preprocessed_image).convert("RGB")

        # Verificar si la imagen contiene texto matemático antes de procesarla
        if not contains_math_content(preprocessed_image):
            logger.info("No se detectó contenido matemático en la imagen")
            return None, None

        # Procesar con el modelo
        pixel_values = feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device)

        # Usar half-precision para acelerar el procesamiento si está disponible
        if device.type == 'cuda':
            model.half()
            pixel_values = pixel_values.half()

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=300,
                num_beams=3,  # Reducido para mayor velocidad
                length_penalty=0.6,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Verificar y limpiar el resultado
        if is_valid_latex(generated_text):
            generated_text = correct_ocr_errors(generated_text)
            generated_text = clean_latex_text(generated_text)
            problem_type = classify_problem_type(generated_text)
            logger.info(f"Fórmula LaTeX detectada: {generated_text}")
            return generated_text, problem_type
        else:
            logger.info("No se detectó una fórmula matemática válida")
            return None, None

    except Exception as e:
        logger.error(f"Error al procesar la imagen con Im2Latex: {e}")
        return None, None


def correct_ocr_errors(text):
    """
    Corrige errores comunes de OCR en las fórmulas detectadas.
    """
    corrections = {
        "—": "-",
        "vb": "√",
        " ": "",
        "\\left(": "(",
        "\\right)": ")",
        "\\left[": "[",
        "\\right]": "]",
        "\\left{": "{",
        "\\right}": "}",
        "\\mathbb{R}": "ℝ",
        "\\mathbb{N}": "ℕ",
        "\\mathbb{Z}": "ℤ",
        "\\oslash": "\\oslash",
        "\\oplus": "\\oplus",
        "\\ominus": "\\ominus",
        "\\otimes": "\\otimes",
        "\\odot": "\\odot",
        "\\bigodot": "\\bigodot",
        "\\bigoplus": "\\bigoplus",
        "\\bigotimes": "\\bigotimes",
        "\\oslash": "\\oslash",
        "\\leq": "\\leq",
        "\\geq": "\\geq",
        "\\neq": "\\neq",
        "\\approx": "\\approx",
        "\\propto": "\\propto",
        "\\infty": "\\infty",
        "\\wedge": "\\wedge",
        "\\vee": "\\vee"
    }
    for wrong_char, correct_char in corrections.items():
        text = text.replace(wrong_char, correct_char)
    return text


def clean_latex_text(latex_text):
    """
    Limpia el texto LaTeX y lo convierte a una forma más legible.
    """
    latex_text = re.sub(r"\s+", " ", latex_text).strip()
    try:
        expr = parse_latex(latex_text)
        clean_latex = latex(expr)
        return clean_latex
    except Exception as e:
        logger.warning(f"Error al limpiar LaTeX: {e}")
        return latex_text  # Devuelve el texto original si hay un error de parsing


def classify_problem_type(latex_formula):
    """
    Clasifica el tipo de problema matemático basado en la fórmula LaTeX.
    """
    types = {
        r"\\int": "Integral",
        r"\\lim": "Límite",
        r"\\frac{d}{d[x-z]}": "Derivada",
        r"\\sum": "Suma",
        r"\\prod": "Producto",
        r"=": "Ecuación",
        r"\\sqrt": "Raíz cuadrada",
        r"\\log": "Logaritmo",
        r"\\sin|\\cos|\\tan": "Trigonometría",
        r"\\matrix": "Matriz",
        r"\\vec": "Vector",
        r"\\infty": "Infinito",
        r"\\in": "Teoría de conjuntos",
        r"\\cup|\\cap": "Operaciones de conjuntos",
        r"\\forall|\\exists": "Lógica matemática",
        r"\\binom": "Combinatoria",
    }
    for pattern, problem_type in types.items():
        if re.search(pattern, latex_formula):
            return problem_type
    return "Expresión algebraica"


def render_latex_to_image(latex_text):
    """
    Renderiza la fórmula LaTeX a una imagen utilizando matplotlib.
    """
    try:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"${latex_text}$", fontsize=20, ha="center", va="center")
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        logger.error(f"Error al renderizar LaTeX a imagen: {e}")
        return None


class Classifier:
    @staticmethod
    def classify_problem(image_path):
        latex_formula, problem_type = process_with_im2latex(cv2.imread(image_path))
        return latex_formula, problem_type

    @staticmethod
    def process_pdf(pdf_path):
        """
        Procesa un archivo PDF y extrae fórmulas matemáticas.
        """
        doc = fitz.open(pdf_path)
        formulas = []

        def process_page(page):
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_path = f"temp/page_{page.number}.png"
            img.save(img_path)
            latex_formula, problem_type = Classifier.classify_problem(img_path)
            os.remove(img_path)
            return latex_formula, problem_type

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_page = {executor.submit(process_page, page): page for page in doc}
            for future in as_completed(future_to_page):
                latex_formula, problem_type = future.result()
                if latex_formula:
                    formulas.append({"formula": latex_formula, "tipo": problem_type})

        return formulas

    @staticmethod
    def process_image(image_path):
        """
        Procesa una imagen y extrae fórmulas matemáticas.
        """
        preprocessed = preprocess_image(cv2.imread(image_path))
        cv2.imwrite("temp/preprocessed.png", preprocessed)
        return Classifier.classify_problem("temp/preprocessed.png")


# Diccionario local optimizado con las expresiones más comunes
latex_equivalentes = {
    "sqrt": "\\sqrt",
    "^": "^{}",
    "pi": "\\pi",
    "inf": "\\infty",
    "sum": "\\sum",
    "int": "\\int",
    "prod": "\\prod",
    "sin": "\\sin",
    "cos": "\\cos",
    "tan": "\\tan",
    "log": "\\log",
    "ln": "\\ln",
    "lim": "\\lim",
    "alpha": "\\alpha",
    "beta": "\\beta",
    "gamma": "\\gamma",
    "delta": "\\delta",
    "epsilon": "\\epsilon",
    "zeta": "\\zeta",
    "eta": "\\eta",
    "theta": "\\theta",
    "iota": "\\iota",
    "kappa": "\\kappa",
    "lambda": "\\lambda",
    "mu": "\\mu",
    "nu": "\\nu",
    "xi": "\\xi",
    "pi": "\\pi",
    "rho": "\\rho",
    "sigma": "\\sigma",
    "tau": "\\tau",
    "upsilon": "\\upsilon",
    "phi": "\\phi",
    "chi": "\\chi",
    "psi": "\\psi",
    "omega": "\\omega",
    "Gamma": "\\Gamma",
    "Delta": "\\Delta",
    "Theta": "\\Theta",
    "Lambda": "\\Lambda",
    "Xi": "\\Xi",
    "Pi": "\\Pi",
    "Sigma": "\\Sigma",
    "Upsilon": "\\Upsilon",
    "Phi": "\\Phi",
    "Psi": "\\Psi",
    "Omega": "\\Omega",
    "in": "\\in",
    "notin": "\\notin",
    "subset": "\\subset",
    "subseteq": "\\subseteq",
    "cup": "\\cup",
    "cap": "\\cap",
    "setminus": "\\setminus",
    "times": "\\times",
    "div": "\\div",
    "pm": "\\pm",
    "mp": "\\mp",
    "cdot": "\\cdot",
    "ast": "\\ast",
    "star": "\\star",
    "circ": "\\circ",
    "bigcirc": "\\bigcirc",
    "oplus": "\\oplus",
    "ominus": "\\ominus",
    "otimes": "\\otimes",
    "oslash": "\\oslash",
    "odot": "\\odot",
    "bigodot": "\\bigodot",
    "bigoplus": "\\bigoplus",
    "bigotimes": "\\bigotimes",
    "bigoplus": "\\bigoplus",
    "bigotimes": "\\bigotimes",
    "leq": "\\leq",
    "geq": "\\geq",
    "neq": "\\neq",
    "approx": "\\approx",
    "propto": "\\propto",
    "infty": "\\infty",
    "wedge": "\\wedge",
    "vee": "\\vee",
    "oplus": "\\oplus",
    "ominus": "\\ominus",
    "otimes": "\\otimes",
    "oslash": "\\oslash"
}


@lru_cache(maxsize=1000)
def buscar_en_api_externa(simbolo):
    """
    Busca un símbolo en una API externa si no está en el diccionario local.

    Args:
        simbolo (str): El símbolo a buscar.

    Returns:
        str: El equivalente LaTeX del símbolo o el símbolo original si no se encuentra.
    """
    url = f"https://api.mathmlcloud.org/convert?input={simbolo}&format=latex"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("latex", simbolo)
    except requests.RequestException as e:
        logger.error(f"Error al buscar símbolo en API externa: {e}")
    return simbolo

def convertir_a_latex(input_text):
    """
    Convierte texto en su equivalente LaTeX.

    Args:
        input_text (str): El texto a convertir.

    Returns:
        str: El texto convertido a formato LaTeX.
    """
    try:
        # Asegúrate de que input_text sea una cadena
        if not isinstance(input_text, str):
            raise ValueError(f"El texto de entrada debe ser una cadena, no {type(input_text)}")

        logger.info(f"Procesando texto: '{input_text}'")

        # Eliminar espacios en blanco innecesarios
        input_text = input_text.strip()

        # Manejar paréntesis
        input_text = input_text.replace("(", "\\left(").replace(")", "\\right)")

        # Manejar potencias
        input_text = re.sub(r'(\d+)\^(\d+)', r'\1^{\2}', input_text)

        # Manejar operaciones básicas
        operaciones = {
            r"(\d+)\s*\+\s*(\d+)": r"\1 + \2",
            r"(\d+)\s*-\s*(\d+)": r"\1 - \2",
            r"(\d+)\s*\*\s*(\d+)": r"\1 \\times \2",
            r"(\d+)\s*/\s*(\d+)": r"\\frac{\1}{\2}",
        }

        for patron, reemplazo in operaciones.items():
            input_text = re.sub(patron, reemplazo, input_text)

        # Manejar funciones matemáticas comunes
        funciones = {
            r"\bsqrt\b": r"\\sqrt",
            r"\bsin\b": r"\\sin",
            r"\bcos\b": r"\\cos",
            r"\btan\b": r"\\tan",
            r"\blog\b": r"\\log",
            r"\bln\b": r"\\ln",
            r"\bexp\b": r"\\exp",
        }

        for patron, reemplazo in funciones.items():
            input_text = re.sub(patron, reemplazo, input_text)

        logger.info(f"Texto convertido: '{input_text}'")
        return input_text

    except Exception as e:
        logger.error(f"Error al procesar el texto '{input_text}': {str(e)}")
        return f"Error: {str(e)}"

def is_math_formula(text, confidence_threshold=0.7):
    """
    Determina si el texto es una fórmula matemática basándose en la presencia de elementos matemáticos.

    Args:
        text (str): El texto a analizar.
        confidence_threshold (float): El umbral de confianza para considerar el texto como fórmula matemática.

    Returns:
        bool: True si el texto es considerado una fórmula matemática, False en caso contrario.
    """
    # Lista de palabras clave y símbolos matemáticos
    math_keywords = [
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "lim",
        "int",
        "sum",
        "prod",
        "frac",
        "sqrt",
    ]
    math_symbols = [
        "+",
        "-",
        "*",
        "/",
        "^",
        "=",
        "<",
        ">",
        "≤",
        "≥",
        "∫",
        "∑",
        "∏",
        "√",
        "∞",
        "π",
    ]

    # Contar ocurrencias de palabras clave y símbolos
    keyword_count = sum(1 for keyword in math_keywords if keyword in text.lower())
    symbol_count = sum(1 for symbol in math_symbols if symbol in text)

    # Calcular la confianza basada en la presencia de elementos matemáticos
    total_elements = len(text.split())
    math_elements = keyword_count + symbol_count
    confidence = math_elements / total_elements if total_elements > 0 else 0

    return confidence >= confidence_threshold

def contains_math_content(image):
    """
    Detecta rápidamente si una imagen contiene contenido matemático.
    """
    try:
        # Detectar características típicas de fórmulas matemáticas
        edges = cv2.Canny(image, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analizar patrones de contornos
        math_patterns = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if 0.1 < circularity < 0.9:  # Típico para símbolos matemáticos
                    math_patterns += 1

        return math_patterns > 5  # Umbral ajustable

    except Exception as e:
        logger.error(f"Error al detectar contenido matemático: {e}")
        return True  # En caso de error, permitir el procesamiento

def is_valid_latex(text):
    """
    Verifica si el texto generado es una fórmula LaTeX válida.
    """
    # Patrones comunes en fórmulas matemáticas
    math_patterns = [
        r'\\[a-zA-Z]+{',  # Comandos LaTeX con llaves
        r'\\[a-zA-Z]+\s',  # Comandos LaTeX con espacio
        r'\$.*\$',         # Contenido entre $
        r'[0-9]+',         # Números
        r'[+\-*/=]',       # Operadores matemáticos
        r'\\frac',         # Fracciones
        r'\\sum',          # Sumas
        r'\\int',          # Integrales
    ]
    
    return any(re.search(pattern, text) for pattern in math_patterns)

def main():
    try:
        logger.info("Módulo de inteligencia artificial iniciado.")
        # Aquí puedes agregar más lógica de inicialización si es necesario
    except Exception as e:
        logger.error(f"Error al iniciar el módulo de IA: {str(e)}")


if __name__ == "__main__":
    main()

