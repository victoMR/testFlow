"""
Módulo de inteligencia artificial para el reconocimiento y clasificación de ecuaciones matemáticas.
Optimizado para rendimiento y precisión en la detección de fórmulas matemáticas.
"""

import logging
import os
import cv2
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import matplotlib.pyplot as plt
import matplotlib
from sympy import latex
from sympy.parsing.latex import parse_latex
from concurrent.futures import ThreadPoolExecutor
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError as RedisTimeoutError
from functools import lru_cache
from dotenv import load_dotenv
from io import BytesIO
import time
import fitz  # PyMuPDF para procesamiento de PDFs
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import glob

# Configuración inicial
load_dotenv()
matplotlib.use("Agg")  # Usar backend no interactivo

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

# Configuración de modelo y dispositivo
model_dir = os.path.join("ai", "Models")
os.makedirs(model_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RedisManager:
    """Gestiona la conexión y operaciones con Redis."""
    
    def __init__(self):
        self.client = self._connect()
        
    def _connect(self):
        """Establece conexión con Redis con reintentos."""
        for _ in range(3):
            try:
                client = Redis(
                    host=os.getenv("REDIS_HOST"),
                    port=int(os.getenv("REDIS_PORT")),
                    db=int(os.getenv("REDIS_DB", 0)),
                    password=os.getenv("REDIS_PASSWORD"),
                    socket_timeout=5,
                    decode_responses=True
                )
                client.ping()
                logger.info("Conexión exitosa a Redis")
                return client
            except (ConnectionError, RedisTimeoutError) as e:
                logger.warning(f"Intento de conexión a Redis fallido: {e}")
        return None

    def get(self, key):
        """Obtiene valor de Redis con manejo de errores."""
        try:
            return self.client.get(key) if self.client else None
        except Exception as e:
            logger.error(f"Error al obtener de Redis: {e}")
            return None

    def set(self, key, value, timeout=3600):
        """Establece valor en Redis con manejo de errores."""
        try:
            if self.client:
                self.client.setex(key, timeout, value)
        except Exception as e:
            logger.error(f"Error al guardar en Redis: {e}")

class ModelManager:
    """Gestiona la carga y operaciones del modelo de IA."""
    
    def __init__(self):
        self.model, self.tokenizer, self.feature_extractor = self._load_model()
        self.model.to(device)
        if device.type == 'cuda':
            self.model.half()
        # Cache para resultados
        self.formula_cache = {}
            
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        logger.info("Cargando modelo de visión...")
        try:
            # Usar modelo más preciso para fórmulas matemáticas
            model = VisionEncoderDecoderModel.from_pretrained(
                "DGurgurov/im2latex",
                cache_dir=model_dir,
                trust_remote_code=True,
                revision="main",
                use_auth_token=False
            )
            
            # Configurar el tokenizer para mejor manejo de fórmulas
            tokenizer = AutoTokenizer.from_pretrained(
                "DGurgurov/im2latex",
                cache_dir=model_dir,
                pad_token="<pad>",
                eos_token="</s>",
                use_fast=True,
                model_max_length=512
            )
            
            # Mejorar el procesamiento de imágenes
            feature_extractor = ViTImageProcessor.from_pretrained(
                "microsoft/swin-base-patch4-window7-224-in22k",
                cache_dir=model_dir,
                do_resize=True,
                size={"height": 224, "width": 224},
                do_normalize=True
            )
            
            logger.info("Modelo cargado exitosamente")
            return model, tokenizer, feature_extractor
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

    def process_formula(self, image, region=None):
        """Procesa una región específica o la imagen completa para extraer fórmulas."""
        try:
            # Si se proporciona una región, recortar la imagen
            if region is not None:
                x, y, w, h = region
                image = image[y:y+h, x:x+w]

            # Preprocesar imagen para el modelo
            image_tensor = self._prepare_image(image)
            
            # Generar predicción
            with torch.no_grad():
                outputs = self.model.generate(
                    image_tensor,
                    max_length=512,
                    num_beams=5,           # Aumentado para mejor búsqueda
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decodificar y obtener scores
            predicted_latex = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            confidence_scores = self._calculate_confidence(outputs)

            # Filtrar y limpiar resultados
            valid_formulas = []
            for latex, score in zip(predicted_latex, confidence_scores):
                if score > 0.5:  # Umbral de confianza
                    cleaned_latex = self._clean_formula(latex)
                    if cleaned_latex:
                        valid_formulas.append({
                            'latex': cleaned_latex,
                            'confidence': score
                        })

            return valid_formulas

        except Exception as e:
            logger.error(f"Error en procesamiento de fórmula: {e}")
            return []

    def _prepare_image(self, image):
        """Prepara la imagen para el modelo."""
        try:
            # Convertir a RGB si es necesario
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convertir a PIL
            pil_image = Image.fromarray(image)

            # Procesar con el feature extractor
            inputs = self.feature_extractor(
                images=pil_image,
                return_tensors="pt"
            )

            return inputs.pixel_values.to(device)

        except Exception as e:
            logger.error(f"Error en preparación de imagen: {e}")
            raise

    def _calculate_confidence(self, outputs):
        """Calcula scores de confianza para las predicciones."""
        try:
            # Calcular probabilidades promedio
            scores = []
            for seq_scores in outputs.scores:
                probs = torch.nn.functional.softmax(seq_scores, dim=-1)
                max_probs = torch.max(probs, dim=-1).values
                avg_prob = torch.mean(max_probs).item()
                scores.append(avg_prob)
            return scores
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return [0.0]

    def _clean_formula(self, latex):
        """Limpia y valida la fórmula LaTeX."""
        try:
            # Eliminar espacios extras
            latex = re.sub(r'\s+', ' ', latex.strip())
            
            # Verificar balance de símbolos
            if not self._check_symbol_balance(latex):
                return None
                
            # Verificar longitud mínima
            if len(latex) < 3:
                return None
                
            # Verificar contenido matemático básico
            if not re.search(r'[+\-*/=\\\{\}\[\]\(\)]', latex):
                return None
                
            return latex
            
        except Exception:
            return None

    def _check_symbol_balance(self, latex):
        """Verifica el balance de símbolos en la fórmula."""
        stack = []
        pairs = {'{': '}', '[': ']', '(': ')'}
        
        for char in latex:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if char != pairs[stack.pop()]:
                    return False
                    
        return len(stack) == 0

class ImageProcessor:
    """Procesa imágenes para detección de fórmulas matemáticas."""
    
    def __init__(self):
        self.last_frame = None
        self.last_result = None
        self.frame_skip = 2
        self.frame_count = 0
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        self._clean_temp_dir()  # Limpiar al inicio
        
    def _clean_temp_dir(self):
        """Limpia completamente el directorio temporal."""
        try:
            files = glob.glob(os.path.join(self.temp_dir, '*'))
            for f in files:
                os.remove(f)
            logger.info("Directorio temporal limpiado")
        except Exception as e:
            logger.error(f"Error limpiando directorio temporal: {e}")

    def _save_debug_image(self, image, stage):
        """Guarda imagen de debug en el directorio temporal."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.temp_dir, f"debug_{stage}_{timestamp}.png")
            cv2.imwrite(filename, image)
            logger.info(f"Imagen de debug guardada: {filename}")
        except Exception as e:
            logger.error(f"Error guardando imagen de debug: {e}")

    def preprocess(self, image):
        """Preprocesamiento mejorado para extracción de fórmulas."""
        try:
            self._clean_temp_dir()  # Limpiar antes de empezar
            
            if not isinstance(image, np.ndarray):
                return None

            # 1. Preprocesamiento inicial
            height, width = image.shape[:2]
            # Mantener aspecto pero asegurar tamaño mínimo
            min_dim = 1000
            scale = max(min_dim / min(height, width), 1.0)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            self._save_debug_image(image, "1_initial")

            # 2. Mejora de contraste adaptativo
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            self._save_debug_image(gray, "2_contrast")

            # 3. Reducción de ruido bilateral
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            self._save_debug_image(denoised, "3_denoised")

            # 4. Detección de bordes múltiple
            edges_canny = cv2.Canny(denoised, 50, 150)
            gradient_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edges_combined = cv2.addWeighted(edges_canny, 0.7, gradient_mag, 0.3, 0)
            self._save_debug_image(edges_combined, "4_edges")

            # 5. Umbralización adaptativa
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                10
            )
            self._save_debug_image(thresh, "5_threshold")

            # 6. Operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            self._save_debug_image(morph, "6_morph")

            # 7. Encontrar y filtrar contornos
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(morph)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Filtrar contornos pequeños
                    cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
            
            # 8. Resultado final
            result = cv2.bitwise_and(morph, mask)
            self._save_debug_image(result, "7_final")

            return result

        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            self._clean_temp_dir()  # Limpiar en caso de error
            return None

    def contains_math(self, image):
        """Detecta si la imagen contiene contenido matemático."""
        try:
            # Verificar similitud con último frame
            if self.last_frame is not None:
                diff = cv2.absdiff(image, self.last_frame)
                if np.mean(diff) < 5.0:
                    return self.last_result

            # Detectar características
            edges = cv2.Canny(image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Análisis de patrones
            math_indicators = 0
            total_area = image.shape[0] * image.shape[1]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < total_area * 0.0001:
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                if 0.1 < aspect_ratio < 10:
                    roi = image[y:y+h, x:x+w]
                    if self._check_symbol_features(roi):
                        math_indicators += 1
                        
                if math_indicators >= 2:
                    self.last_frame = image.copy()
                    self.last_result = True
                    return True
            
            self.last_frame = image.copy()
            self.last_result = False
            return False
            
        except Exception as e:
            logger.error(f"Error en detección de contenido matemático: {e}")
            return False

    def _check_symbol_features(self, roi):
        """Verificación mejorada de características matemáticas."""
        try:
            # Normalizar y preparar ROI
            roi = cv2.resize(roi, (64, 64))
            
            # 1. Análisis de gradientes
            gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            
            # 2. Características geométricas
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
                
            # Análisis del contorno principal
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                return False
            
            # Características de forma
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                return False
            solidity = float(area) / hull_area
            
            # 3. Análisis de densidad y distribución
            pixel_density = np.sum(roi > 0) / roi.size
            
            # 4. Análisis de simetría
            roi_flipped = cv2.flip(roi, 1)
            symmetry_score = np.sum(roi == roi_flipped) / roi.size
            
            # Criterios combinados para símbolos matemáticos
            is_math_symbol = (
                (0.1 < circularity < 0.9) and  # No demasiado circular ni irregular
                (0.3 < solidity < 0.95) and    # Forma coherente pero no demasiado simple
                (0.1 < pixel_density < 0.7) and # Densidad típica de símbolos
                (symmetry_score > 0.7)          # Cierto grado de simetría
            )
            
            return is_math_symbol
            
        except Exception as e:
            logger.error(f"Error en análisis de símbolos: {e}")
            return False

class LatexProcessor:
    """Procesa y valida fórmulas LaTeX."""
    
    @staticmethod
    def clean_latex(text):
        """Limpia y formatea texto LaTeX."""
        try:
            text = re.sub(r"\s+", " ", text).strip()
            expr = parse_latex(text)
            return latex(expr)
        except Exception as e:
            logger.warning(f"Error al limpiar LaTeX: {e}")
            return text

    @staticmethod
    def convert_to_texjs(latex_text):
        """Convierte LaTeX a formato compatible con TeXJS."""
        try:
            # Diccionario de conversiones para TeXJS
            texjs_conversions = {
                r'\\left': '',  # Eliminar \left
                r'\\right': '', # Eliminar \right
                r'\\mathrm{d}': 'd', # Simplificar diferencial
                r'\\mathbb{R}': '\\R', # Símbolos especiales
                r'\\mathbb{N}': '\\N',
                r'\\mathbb{Z}': '\\Z',
                r'\\begin{array}': '\\begin{matrix}', # Matrices
                r'\\end{array}': '\\end{matrix}',
                r'\\text': '\\mathrm', # Texto en fórmulas
                r'\\operatorname': '\\mathrm', # Operadores
                r'\\displaystyle': '', # Eliminar estilo display
            }
            
            # Aplicar conversiones
            result = latex_text
            for old, new in texjs_conversions.items():
                result = re.sub(old, new, result)
            
            # Ajustar fracciones
            result = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'\\frac{\1}{\2}', result)
            
            # Ajustar subíndices y superíndices
            result = re.sub(r'_([^{])', r'_{\\1}', result)
            result = re.sub(r'\^([^{])', r'^{\\1}', result)
            
            # Ajustar espaciado
            result = re.sub(r'\\,', ' ', result)
            result = re.sub(r'\\;', ' ', result)
            result = re.sub(r'\\:', ' ', result)
            result = re.sub(r'\\!', '', result)
            
            logger.info(f"LaTeX convertido para TeXJS: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error al convertir LaTeX para TeXJS: {e}")
            return latex_text

    @staticmethod
    def is_valid_latex(text):
        """Verifica si el texto es una fórmula LaTeX válida."""
        patterns = [
            r'\\[a-zA-Z]+{',
            r'\\[a-zA-Z]+\s',
            r'\$.*\$',
            r'[0-9]+',
            r'[+\-*/=]',
            r'\\frac',
            r'\\sum',
            r'\\int',
            r'\\sqrt',
            r'\\lim',
            r'\\infty',
            r'\\alpha|\\beta|\\gamma|\\delta',
            r'\\sin|\\cos|\\tan',
            r'\\log|\\ln',
            r'\\partial',
            r'\\nabla',
            r'\\vec',
            r'\\matrix',
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def classify_problem(latex_formula):
        """Clasifica el tipo de problema matemático."""
        types = {
            r"\\int": "Integral",
            r"\\lim": "Límite",
            r"\\frac{d}{d[x-z]}": "Derivada",
            r"\\sum": "Suma",
            r"=": "Ecuación",
            r"\\sqrt": "Raíz",
            r"\\log": "Logaritmo",
            r"\\sin|\\cos|\\tan": "Trigonometría",
            r"\\matrix": "Matriz",
            r"\\vec": "Vector",
            r"\\alpha|\\beta|\\gamma|\\delta": "Letras griegas",
            r"\\pi": "Número pi",
            r"\\infty": "Infinito",
            r"\\partial": "Derivada parcial",
            r"\\nabla": "Gradiente",
            r"\\mathrm": "Función matemática",
            r"\\text": "Texto",
            r"\\operatorname": "Operador",
            r"\\begin{matrix}": "Matriz",
            r"\\end{matrix}": "Matriz",
            r"\\begin{sum}": "Sumatoria",
            r"\\end{sum}": "Sumatoria"
        }
        
        for pattern, problem_type in types.items():
            if re.search(pattern, latex_formula):
                return problem_type
        return "Expresión algebraica"

class FormulaDetector:
    """Clase principal para detección y procesamiento de fórmulas."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        self.latex_processor = LatexProcessor()
        self.frame_buffer = []
        self.buffer_size = 5
        self.last_processed_time = 0
        self.min_process_interval = 0.2
        self.confidence_threshold = 0.35
        # Crear directorio temporal si no existe
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

    def _save_debug_image(self, image, stage):
        """Guarda imagen de debug en el directorio temporal."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.temp_dir, f"debug_{stage}_{timestamp}.png")
            cv2.imwrite(filename, image)
            logger.info(f"Imagen de debug guardada: {filename}")
        except Exception as e:
            logger.error(f"Error guardando imagen de debug: {e}")

    def _clean_temp_dir(self):
        """Limpia el directorio temporal."""
        try:
            files = glob.glob(os.path.join(self.temp_dir, '*'))
            for f in files:
                os.remove(f)
            logger.info("Directorio temporal limpiado")
        except Exception as e:
            logger.error(f"Error limpiando directorio temporal: {e}")

    def _calculate_math_score(self, image):
        """Calcula un score de confianza para contenido matemático."""
        try:
            self._clean_temp_dir()  # Limpiar antes de empezar
            
            # Guardar imagen original para debug
            self._save_debug_image(image, "original")
            
            # Preprocesar imagen para análisis
            blur = cv2.GaussianBlur(image, (5,5), 0)
            self._save_debug_image(blur, "blur")
            
            edges = cv2.Canny(blur, 30, 150)
            self._save_debug_image(edges, "edges")
            
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            total_area = image.shape[0] * image.shape[1]
            valid_symbols = 0
            total_symbols = 0
            
            # Crear imagen para visualizar contornos
            debug_contours = image.copy()
            
            # Analizar jerarquía de contornos
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < total_area * 0.00001:  # Umbral más bajo
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                # Criterios más flexibles
                if 0.05 < aspect_ratio < 15:  # Rango más amplio
                    roi = image[y:y+h, x:x+w]
                    total_symbols += 1
                    
                    if self.image_processor._check_symbol_features(roi):
                        valid_symbols += 1
                        # Dibujar rectángulo verde para símbolos válidos
                        cv2.rectangle(debug_contours, (x,y), (x+w,y+h), (0,255,0), 2)
                    else:
                        # Dibujar rectángulo rojo para símbolos no válidos
                        cv2.rectangle(debug_contours, (x,y), (x+w,y+h), (0,0,255), 1)
            
            # Guardar imagen con contornos detectados
            self._save_debug_image(debug_contours, "detected_symbols")
            
            # Calcular score
            if total_symbols == 0:
                return 0.0
                
            symbol_ratio = valid_symbols / total_symbols
            coverage = sum(cv2.contourArea(cnt) for cnt in contours) / total_area
            
            # Score combinado
            score = (symbol_ratio * 0.7 + coverage * 0.3)
            logger.info(f"Score matemático: {score:.4f} (símbolos: {valid_symbols}/{total_symbols}, coverage: {coverage:.4f})")
            
            # Guardar imagen final con score
            score_img = debug_contours.copy()
            cv2.putText(score_img, f"Score: {score:.4f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            self._save_debug_image(score_img, "final_score")
            
            return score
            
        except Exception as e:
            logger.error(f"Error al calcular score matemático: {e}")
            self._clean_temp_dir()  # Limpiar en caso de error
            return 0.0

    def process_image(self, image):
        """Detecta y extrae fórmulas matemáticas."""
        try:
            # Preprocesar imagen
            processed_image = self.image_processor.preprocess(image)
            if processed_image is None:
                return None

            # Detectar regiones con posibles fórmulas
            regions = self._detect_formula_regions(processed_image)
            
            # Procesar cada región
            formulas = []
            for region in regions:
                results = self.model_manager.process_formula(processed_image, region)
                formulas.extend(results)

            # Si no se encontraron regiones, procesar la imagen completa
            if not formulas:
                results = self.model_manager.process_formula(processed_image)
                formulas.extend(results)

            # Ordenar por confianza y eliminar duplicados
            formulas.sort(key=lambda x: x['confidence'], reverse=True)
            unique_formulas = []
            seen = set()
            for formula in formulas:
                if formula['latex'] not in seen:
                    seen.add(formula['latex'])
                    unique_formulas.append(formula)

            return unique_formulas if unique_formulas else None

        except Exception as e:
            logger.error(f"Error en procesamiento de imagen: {e}")
            return None

    def _verify_math_content(self, image):
        """Verificación más estricta de contenido matemático."""
        try:
            # Verificar presencia de personas
            if self._detect_persons(image):
                logger.info("Se detectó una persona en la imagen")
                return False
                
            # Verificar características matemáticas
            math_score = self._calculate_math_score(image)
            
            # Definir rangos válidos para scores matemáticos
            MIN_SCORE = 0.01  # Score mínimo aceptable
            MAX_SCORE = 0.99  # Score máximo aceptable
            
            # Verificar si el score está en el rango válido
            is_math = MIN_SCORE <= math_score <= MAX_SCORE
            
            if is_math:
                logger.info(f"Contenido matemático detectado con score válido: {math_score}")
            else:
                if math_score < MIN_SCORE:
                    logger.info(f"Score muy bajo para ser fórmula: {math_score}")
                elif math_score > MAX_SCORE:
                    logger.info(f"Score demasiado alto, posible falso positivo: {math_score}")
                else:
                    logger.info(f"No se detectó contenido matemático. Score: {math_score}")
                    
            return is_math
                
        except Exception as e:
            logger.error(f"Error en verificación de contenido: {e}")
            return False

    def _detect_persons(self, image):
        """Detecta si hay personas en la imagen."""
        # Aquí puedes implementar un detector de personas simple
        # Por ahora, usaremos una implementación básica
        return False

    def render_latex(self, latex_text):
        """Renderiza fórmula LaTeX a imagen."""
        try:
            plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, f"${latex_text}$", 
                    fontsize=20, ha="center", va="center")
            plt.axis("off")
            
            buf = BytesIO()
            plt.savefig(buf, format="png", 
                       bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close()
            
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            logger.error(f"Error al renderizar LaTeX: {e}")
            return None

    def convertir_a_latex(self, texto):
        """Convierte texto plano a formato LaTeX."""
        try:
            # Diccionario de conversiones matemáticas comunes
            conversiones = {
                # Operadores básicos
                '+': '+',
                '-': '-',
                '*': '\\times',
                '/': '\\div',
                
                # Funciones matemáticas
                'sqrt': '\\sqrt',
                'sin': '\\sin',
                'cos': '\\cos',
                'tan': '\\tan',
                'log': '\\log',
                'ln': '\\ln',
                'lim': '\\lim',
                
                # Símbolos especiales
                'pi': '\\pi',
                'inf': '\\infty',
                'alpha': '\\alpha',
                'beta': '\\beta',
                'gamma': '\\gamma',
                'delta': '\\delta',
                'sum': '\\sum',
                'int': '\\int',
                'prod': '\\prod',
                
                # Relaciones
                '>=': '\\geq',
                '<=': '\\leq',
                '!=': '\\neq',
                '~=': '\\approx',
                
                # Otros símbolos
                'partial': '\\partial',
                'nabla': '\\nabla',
                'grad': '\\nabla',
                'div': '\\nabla \\cdot',
                'curl': '\\nabla \\times',
            }

            # Patrones para detectar expresiones matemáticas comunes
            patrones = [
                # Fracciones (a/b)
                (r'(\d+)/(\d+)', r'\\frac{\1}{\2}'),
                
                # Potencias (a^b)
                (r'(\w+)\^(\w+)', r'{\1}^{\2}'),
                
                # Subíndices (a_b)
                (r'(\w+)_(\w+)', r'{\1}_{\2}'),
                
                # Raíces cuadradas
                (r'sqrt\((.*?)\)', r'\\sqrt{\1}'),
                
                # Funciones trigonométricas
                (r'sin\((.*?)\)', r'\\sin(\1)'),
                (r'cos\((.*?)\)', r'\\cos(\1)'),
                (r'tan\((.*?)\)', r'\\tan(\1)'),
                
                # Logaritmos
                (r'log\((.*?)\)', r'\\log(\1)'),
                (r'ln\((.*?)\)', r'\\ln(\1)'),
                
                # Límites
                (r'lim_(\w+)->(\w+)', r'\\lim_{\1 \\to \2}'),
                
                # Integrales
                (r'int_(\w+)\^(\w+)', r'\\int_{\1}^{\2}'),
                
                # Sumatorias
                (r'sum_(\w+)\^(\w+)', r'\\sum_{\1}^{\2}'),
            ]

            # Aplicar conversiones directas
            resultado = texto
            for original, latex in conversiones.items():
                resultado = resultado.replace(original, latex)

            # Aplicar patrones más complejos
            for patron, reemplazo in patrones:
                resultado = re.sub(patron, reemplazo, resultado)

            # Limpiar espacios extra
            resultado = re.sub(r'\s+', ' ', resultado).strip()

            # Convertir a formato compatible con TeXJS
            resultado = self.latex_processor.convert_to_texjs(resultado)

            logger.info(f"Texto convertido a LaTeX: {resultado}")
            return resultado

        except Exception as e:
            logger.error(f"Error al convertir texto a LaTeX: {e}")
            return texto

    def process_pdf(self, pdf_path):
        """
        Procesa un archivo PDF y extrae fórmulas matemáticas con alta precisión.
        """
        try:
            logger.info(f"Iniciando procesamiento de PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            # Verificar límites de páginas
            if len(doc) > 50:
                raise ValueError("El PDF excede el límite de 50 páginas")
            if len(doc) < 1:
                raise ValueError("El PDF debe tener al menos 1 página")

            # Configurar procesamiento paralelo
            max_workers = min(os.cpu_count(), len(doc))
            formulas_detectadas = []
            processed_pages = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Procesar páginas en paralelo
                future_to_page = {
                    executor.submit(self._process_pdf_page, doc[page_num], page_num): page_num 
                    for page_num in range(len(doc))
                }
                
                # Recolectar resultados manteniendo el orden
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        page_formulas = future.result()
                        if page_formulas:
                            formulas_detectadas.extend(page_formulas)
                        processed_pages += 1
                        logger.info(f"Procesada página {page_num + 1}/{len(doc)}")
                    except Exception as e:
                        logger.error(f"Error en página {page_num + 1}: {e}")

            # Filtrar y validar resultados
            formulas_validadas = self._validate_formulas(formulas_detectadas)
            
            logger.info(f"Procesamiento de PDF completado. "
                       f"Fórmulas encontradas: {len(formulas_validadas)}")
            
            return formulas_validadas

        except Exception as e:
            logger.error(f"Error en procesamiento de PDF: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()

    def _process_pdf_page(self, page, page_num):
        """
        Procesa una página individual del PDF.
        """
        try:
            # Extraer imágenes y texto de la página
            formulas_en_pagina = []
            
            # Procesar imágenes en la página
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = self._extract_image(page.parent, xref)
                    if base_image:
                        formula = self.process_image(base_image)
                        if formula:
                            formulas_en_pagina.append({
                                'formula': formula,
                                'tipo': self.latex_processor.classify_problem(formula),
                                'pagina': page_num + 1,
                                'confidence': self._calculate_confidence(formula),
                                'origen': 'imagen'
                            })
                except Exception as e:
                    logger.error(f"Error procesando imagen {img_index} en página {page_num + 1}: {e}")

            # Procesar texto en la página
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                try:
                    text = block[4]
                    if self._is_potential_formula(text):
                        formula = self._extract_formula_from_text(text)
                        if formula:
                            formulas_en_pagina.append({
                                'formula': formula,
                                'tipo': self.latex_processor.classify_problem(formula),
                                'pagina': page_num + 1,
                                'confidence': self._calculate_confidence(formula),
                                'origen': 'texto'
                            })
                except Exception as e:
                    logger.error(f"Error procesando bloque de texto en página {page_num + 1}: {e}")

            return formulas_en_pagina

        except Exception as e:
            logger.error(f"Error procesando página {page_num + 1}: {e}")
            return []

    def _extract_image(self, doc, xref):
        """
        Extrae y preprocesa una imagen del PDF.
        """
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # Convertir a formato numpy
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, -1)
            
            return img_array
        except Exception as e:
            logger.error(f"Error extrayendo imagen: {e}")
            return None

    def _is_potential_formula(self, text):
        """
        Determina si un bloque de texto podría contener una fórmula matemática.
        """
        math_indicators = [
            r'\$', r'\[', r'\(', r'\\frac', r'\\sum', r'\\int',
            r'\\alpha', r'\\beta', r'=', r'\+', r'-', r'\*', r'/',
            r'^', r'_', r'\\sqrt', r'\\lim', r'\\infty'
        ]
        return any(indicator in text for indicator in math_indicators)

    def _extract_formula_from_text(self, text):
        """
        Extrae y valida fórmulas matemáticas del texto.
        """
        try:
            # Buscar patrones de fórmulas
            formula_patterns = [
                r'\$(.+?)\$',
                r'\\\[(.+?)\\\]',
                r'\\\((.+?)\\\)',
            ]
            
            for pattern in formula_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    formula = match.group(1)
                    if self.latex_processor.is_valid_latex(formula):
                        return self.latex_processor.clean_latex(formula)
            return None
        except Exception as e:
            logger.error(f"Error extrayendo fórmula de texto: {e}")
            return None

    def _calculate_confidence(self, formula):
        """
        Calcula el nivel de confianza de la fórmula detectada.
        """
        try:
            # Factores que aumentan la confianza
            confidence = 0.0
            
            # Complejidad de la fórmula
            if len(formula) > 10:
                confidence += 0.2
            
            # Presencia de símbolos matemáticos comunes
            math_symbols = [r'\\frac', r'\\sum', r'\\int', r'\\sqrt']
            for symbol in math_symbols:
                if symbol in formula:
                    confidence += 0.15
            
            # Estructura balanceada
            if self._check_balanced_structure(formula):
                confidence += 0.2
            
            # Validación sintáctica
            if self.latex_processor.is_valid_latex(formula):
                confidence += 0.3
            
            return min(confidence * 100, 100)  # Convertir a porcentaje
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.0

    def _check_balanced_structure(self, formula):
        """
        Verifica que la estructura de la fórmula esté balanceada.
        """
        try:
            stack = []
            brackets = {'{': '}', '[': ']', '(': ')'}
            
            for char in formula:
                if char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return False
                    if char != brackets[stack.pop()]:
                        return False
            
            return len(stack) == 0
        except Exception:
            return False

    def _validate_formulas(self, formulas):
        """
        Filtra y valida las fórmulas detectadas.
        """
        validated_formulas = []
        seen_formulas = set()
        
        for formula_data in formulas:
            formula = formula_data['formula']
            confidence = formula_data['confidence']
            
            # Evitar duplicados
            if formula in seen_formulas:
                continue
                
            # Verificar confianza mínima
            if confidence < 90:
                continue
                
            # Validación adicional
            if (self.latex_processor.is_valid_latex(formula) and 
                self._check_balanced_structure(formula)):
                validated_formulas.append(formula_data)
                seen_formulas.add(formula)
        
        return validated_formulas

# Inicialización del detector
formula_detector = FormulaDetector()

