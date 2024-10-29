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
            
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        logger.info("Cargando modelo de visión...")
        try:
            model = VisionEncoderDecoderModel.from_pretrained(
                "DGurgurov/im2latex", 
                cache_dir=model_dir,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "DGurgurov/im2latex",
                cache_dir=model_dir,
                pad_token="<pad>",
                eos_token="</s>"
            )
            feature_extractor = ViTImageProcessor.from_pretrained(
                "microsoft/swin-base-patch4-window7-224-in22k",
                cache_dir=model_dir
            )
            logger.info("Modelo cargado exitosamente")
            return model, tokenizer, feature_extractor
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

class ImageProcessor:
    """Procesa imágenes para detección de fórmulas matemáticas."""
    
    def __init__(self):
        # Cache para resultados de procesamiento
        self.last_frame = None
        self.last_result = None
        self.frame_skip = 2  # Procesar cada N frames
        self.frame_count = 0
        
    def preprocess(self, image):
        """Preprocesa la imagen para mejorar la detección."""
        try:
            if not isinstance(image, np.ndarray):
                return None
            
            # Redimensionar imagen para procesamiento más rápido
            height, width = image.shape[:2]
            max_dimension = 800
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            # Convertir a escala de grises y mejorar contraste
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Aplicar umbralización adaptativa
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Eliminar ruido manteniendo detalles importantes
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return None

    def contains_math(self, image):
        """Detecta si la imagen contiene contenido matemático de manera eficiente."""
        try:
            # Verificar si es similar al último frame procesado
            if self.last_frame is not None:
                diff = cv2.absdiff(image, self.last_frame)
                if np.mean(diff) < 5.0:  # Umbral de diferencia
                    return self.last_result

            # Detectar características específicas de fórmulas matemáticas
            edges = cv2.Canny(image, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Análisis rápido de patrones matemáticos
            math_indicators = 0
            total_area = image.shape[0] * image.shape[1]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < total_area * 0.0001:  # Ignorar contornos muy pequeños
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                # Características típicas de símbolos matemáticos
                if 0.2 < aspect_ratio < 5:
                    roi = image[y:y+h, x:x+w]
                    if self._check_symbol_features(roi):
                        math_indicators += 1
                        
                if math_indicators >= 3:  # Umbral mínimo de indicadores
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
        """Verifica características específicas de símbolos matemáticos."""
        try:
            # Calcular histograma de gradientes
            gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            
            # Verificar distribución de gradientes típica de símbolos matemáticos
            hist = np.histogram(ang, bins=8)[0]
            hist = hist / np.sum(hist)
            
            # Patrones típicos de símbolos matemáticos
            vertical_lines = hist[0] + hist[4]  # 0° y 180°
            horizontal_lines = hist[2] + hist[6]  # 90° y 270°
            diagonals = hist[1] + hist[3] + hist[5] + hist[7]  # Diagonales
            
            return (vertical_lines > 0.2 or horizontal_lines > 0.2 or diagonals > 0.3)
            
        except Exception:
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
        self.confidence_threshold = 0.35  # Reducido de 0.8 a 0.35
        
    def process_image(self, image):
        """Solo detecta y extrae fórmulas matemáticas."""
        try:
            current_time = time.time()
            if current_time - self.last_processed_time < self.min_process_interval:
                return None
                
            # Preprocesar imagen
            processed_image = self.image_processor.preprocess(image)
            if processed_image is None:
                return None
                
            # Verificación de contenido matemático con umbral más bajo
            if not self._verify_math_content(processed_image):
                return None
                
            # Procesar con el modelo
            pil_image = Image.fromarray(processed_image).convert("RGB")
            pixel_values = self.model_manager.feature_extractor(
                images=pil_image,
                return_tensors="pt"
            ).pixel_values.to(device)
            
            attention_mask = torch.ones(
                (pixel_values.shape[0], pixel_values.shape[2] * pixel_values.shape[3]),
                device=device
            )
            
            with torch.no_grad():
                outputs = self.model_manager.model.generate(
                    pixel_values,
                    attention_mask=attention_mask,
                    max_length=300,
                    num_beams=2,          # Reducido de 3 a 2 para mayor velocidad
                    length_penalty=0.5,    # Reducido de 0.6 a 0.5
                    early_stopping=True
                )
                
            latex_text = self.model_manager.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            
            # Ser más permisivo con la validación de LaTeX
            if not self.latex_processor.is_valid_latex(latex_text):
                return None
                
            cleaned_latex = self.latex_processor.clean_latex(latex_text)
            logger.info(f"Fórmula LaTeX detectada: {cleaned_latex}")
            return cleaned_latex
            
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
            is_math = math_score >= self.confidence_threshold
            
            if is_math:
                logger.info(f"Contenido matemático detectado con score: {math_score}")
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

    def _calculate_math_score(self, image):
        """Calcula un score de confianza para contenido matemático."""
        try:
            edges = cv2.Canny(image, 50, 150)  # Reducidos los umbrales de detección de bordes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            math_indicators = 0
            total_area = image.shape[0] * image.shape[1]
            valid_symbols = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Reducir el umbral de área mínima
                if area < total_area * 0.00005:  # Cambiado de 0.0001 a 0.00005
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                
                # Ampliar el rango de proporciones aceptables
                if 0.1 < aspect_ratio < 10:  # Cambiado de 0.2-5 a 0.1-10
                    roi = image[y:y+h, x:x+w]
                    if self.image_processor._check_symbol_features(roi):
                        valid_symbols += 1
            
            score = valid_symbols / max(len(contours), 1)
            logger.info(f"Score matemático calculado: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error al calcular score matemático: {e}")
            return 0.0

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

