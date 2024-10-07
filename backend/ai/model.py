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
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sympy import sympify, latex
from sympy.parsing.latex import parse_latex
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from functools import lru_cache

# Configuración del logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ruta de la carpeta para los modelos
model_dir = os.path.join("ai", "Models")
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
    """
    try:
        logging.info("Preprocesando la imagen")

        # Ensure the image is in the correct format
        if isinstance(image, np.ndarray):
            # If it's already a numpy array, use it directly
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        else:
            logging.error("La imagen no es un array de numpy válido")
            return None

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Aplicar umbral adaptativo
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Aplicar operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Mejorar el contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(morph)

        logging.info("Preprocesamiento de imagen completado.")
        return contrast
    except Exception as e:
        logging.error(f"Error en el preprocesamiento de la imagen: {e}")
        return None


def process_with_im2latex(image):
    """
    Procesa la imagen usando el modelo Im2Latex y devuelve la fórmula detectada.
    """
    try:
        logging.info("Procesando imagen con el modelo Im2Latex")
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            raise ValueError("Error en el preprocesamiento de la imagen")

        image = Image.fromarray(preprocessed_image).convert("RGB")

        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                attention_mask=attention_mask,
                max_length=300,
                num_beams=5,
                length_penalty=0.6,
                no_repeat_ngram_size=2,
            )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        generated_text = correct_ocr_errors(generated_text)
        generated_text = clean_latex_text(generated_text)
        problem_type = classify_problem_type(generated_text)

        logging.info(f"Fórmula LaTeX detectada: {generated_text}")
        logging.info(f"Tipo de problema: {problem_type}")
        
        return generated_text, problem_type
    except Exception as e:
        logging.error(f"Error al procesar la imagen con Im2Latex: {e}")
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
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\delta": "δ",
        "\\epsilon": "ε",
        "\\zeta": "ζ",
        "\\eta": "η",
        "\\theta": "θ",
        "\\iota": "ι",
        "\\kappa": "κ",
        "\\lambda": "λ",
        "\\mu": "μ",
        "\\nu": "ν",
        "\\xi": "ξ",
        "\\pi": "π",
        "\\rho": "ρ",
        "\\sigma": "σ",
        "\\tau": "τ",
        "\\upsilon": "υ",
        "\\phi": "φ",
        "\\chi": "χ",
        "\\psi": "ψ",
        "\\omega": "ω",
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
        logging.error(f"Error al limpiar LaTeX: {e}")
        return latex_text


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
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"${latex_text}$", fontsize=20, ha="center", va="center")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        buf.close()
        return img
    except Exception as e:
        logging.error(f"Error al renderizar LaTeX a imagen: {e}")
        return None


class Classifier:
    @staticmethod
    def classify_problem(image_path):
        latex_formula = process_with_im2latex(image_path)
        if latex_formula:
            corrected_formula = correct_ocr_errors(latex_formula)
            cleaned_formula = clean_latex_text(corrected_formula)
            problem_type = classify_problem_type(cleaned_formula)
            return cleaned_formula, problem_type
        return "", "Desconocido"

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
        preprocessed = preprocess_image(image_path)
        cv2.imwrite("temp/preprocessed.png", preprocessed)
        return Classifier.classify_problem("temp/preprocessed.png")


def main():
    logging.info("Módulo de inteligencia artificial iniciado.")


if __name__ == "__main__":
    main()