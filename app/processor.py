import os
import logging
import boto3
import PyPDF2
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from regex import regex  # Para búsqueda aproximada
import fitz  # PyMuPDF para mejor manejo de PDFs
from sqlalchemy import create_engine, text, and_, or_
from config import processing_config, LOGGING_CONFIG
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import spacy
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
import io
import numpy as np
from great_expectations.dataset import PandasDataset
from elasticsearch import Elasticsearch
from dataclasses import dataclass
from enum import Enum

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class EstadoProcesamiento(Enum):
    PENDIENTE = "PENDIENTE"
    EN_PROCESO = "EN_PROCESO"
    COMPLETADO = "COMPLETADO"
    ERROR = "ERROR"
    REQUIERE_REVISION = "REQUIERE_REVISION"

@dataclass
class ErrorValidacion:
    tipo: str
    descripcion: str
    severidad: str
    contexto: Dict[str, Any]
    timestamp: datetime = datetime.now()

class ValidadorDocumento:
    def __init__(self, engine):
        self.engine = engine
        self.errores: List[ErrorValidacion] = []
        
    def validar_estructura_jerarquica(self, documento: Dict) -> bool:
        """Valida la estructura jerárquica del documento"""
        try:
            # Validar secuencia de títulos
            if documento.get('titulos'):
                numeros_titulo = [self._convertir_romano_a_numero(t['numero']) for t in documento['titulos']]
                if not all(a < b for a, b in zip(numeros_titulo, numeros_titulo[1:])):
                    self.errores.append(ErrorValidacion(
                        tipo="ERROR_SECUENCIA",
                        descripcion="Secuencia de títulos incorrecta",
                        severidad="ALTA",
                        contexto={"titulos": documento['titulos']}
                    ))
                    return False
            
            # Validar que cada artículo pertenezca a un capítulo
            if documento.get('articulos'):
                for articulo in documento['articulos']:
                    if not self._validar_pertenencia_articulo(articulo, documento):
                        return False
            
            return True
            
        except Exception as e:
            self.errores.append(ErrorValidacion(
                tipo="ERROR_VALIDACION",
                descripcion=f"Error en validación de estructura: {str(e)}",
                severidad="ALTA",
                contexto={"documento_id": documento.get('id')}
            ))
            return False
    
    def _convertir_romano_a_numero(self, romano: str) -> int:
        """Convierte número romano a entero"""
        valores = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        numero = 0
        valor_anterior = 0
        
        for letra in reversed(romano.upper()):
            valor = valores[letra]
            if valor >= valor_anterior:
                numero += valor
            else:
                numero -= valor
            valor_anterior = valor
            
        return numero
    
    def _validar_pertenencia_articulo(self, articulo: Dict, documento: Dict) -> bool:
        """Valida que un artículo pertenezca a un capítulo válido"""
        # Implementar lógica de validación
        return True

class NormaProcessor:
    def __init__(self):
        """Inicializa el procesador de normas legales"""
        self.s3_client = boto3.client('s3')
        self.engine = create_engine(
            f'postgresql://{processing_config.db_user}:{processing_config.db_password}@'
            f'{processing_config.db_host}:{processing_config.db_port}/{processing_config.db_name}'
        )
        
        # Inicializar procesadores especializados
        self.pdf_processor = PDFProcessor()
        self.texto_processor = TextoLegalProcessor()
        self.validador = ValidadorDocumento(self.engine)
        self.quality_validator = QualityValidator(self.engine)
        
        # Inicializar cliente de Claude y otros componentes
        self.claude = anthropic.Client(api_key=processing_config.anthropic_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
    def process_document(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Procesa un documento PDF completo
        """
        try:
            # Descargar PDF de S3
            local_path = f"/tmp/{os.path.basename(key)}"
            self.s3_client.download_file(bucket, key, local_path)
            
            # Extraer texto y metadata del PDF
            texto = self.pdf_processor.extract_text_from_pdf(local_path)
            pdf_metadata = self.pdf_processor.extract_metadata_from_pdf(local_path)
            
            # Extraer estructura del texto
            estructura = self.texto_processor.extract_structure(texto)
            
            # Procesar texto largo para clasificación
            resultados_claude = self.procesar_texto_largo(texto)
            
            # Construir documento completo
            documento = self._construir_documento(
                texto=texto,
                estructura=estructura,
                pdf_metadata=pdf_metadata,
                resultados_claude=resultados_claude,
                nombre_archivo=os.path.basename(key)
            )
            
            # Validar estructura jerárquica
            if not self.validador.validar_estructura_jerarquica(documento):
                for error in self.validador.errores:
                    logger.warning(f"Error de validación: {error.descripcion}")
                documento['estado'] = EstadoProcesamiento.REQUIERE_REVISION.value
            
            # Validar calidad del documento
            es_valido, errores = self.quality_validator.validate_document(documento)
            if not es_valido:
                for error in errores:
                    logger.warning(f"Error de calidad: {error.descripcion}")
                documento['estado'] = EstadoProcesamiento.ERROR.value
            
            # Guardar en base de datos
            self._guardar_documento(documento)
            
            # Limpiar archivos temporales
            os.remove(local_path)
            
            return documento
            
        except Exception as e:
            logger.error(f"Error procesando documento {key}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
    
    def _construir_documento(self, texto: str, estructura: Dict, pdf_metadata: Dict,
                           resultados_claude: List[Dict], nombre_archivo: str) -> Dict:
        """
        Construye el documento final con toda la información procesada
        """
        # Obtener categoría principal y palabras clave
        categoria_principal = None
        palabras_clave = set()
        
        for resultado in resultados_claude:
            if resultado.get('categoria_principal'):
                if not categoria_principal:
                    categoria_principal = resultado['categoria_principal']
                palabras_clave.update(resultado.get('palabras_clave', []))
        
        # Construir documento
        documento = {
            'id': f"NORMA_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'categoria_principal': categoria_principal,
            'titulo': estructura['titulos'][0]['nombre'] if estructura['titulos'] else None,
            'texto_norma': texto,
            'palabras_clave': list(palabras_clave)[:10],
            'nombre_archivo': nombre_archivo,
            'fecha_extraccion': datetime.now().date(),
            'estado': EstadoProcesamiento.PENDIENTE.value,
            'metadata_pdf': pdf_metadata,
            'estructura': estructura
        }
        
        # Agregar artículos
        documento['articulos'] = []
        for articulo in estructura.get('articulos', []):
            documento['articulos'].append({
                'numero': articulo['numero'],
                'contenido': articulo['contenido'],
                'titulo_padre': self._encontrar_titulo_padre(articulo, estructura),
                'capitulo_padre': self._encontrar_capitulo_padre(articulo, estructura)
            })
        
        return documento
    
    def _encontrar_titulo_padre(self, articulo: Dict, estructura: Dict) -> Optional[str]:
        """
        Encuentra el título al que pertenece un artículo
        """
        for titulo in reversed(estructura.get('titulos', [])):
            if articulo['inicio'] > titulo['inicio']:
                return titulo['numero']
        return None
    
    def _encontrar_capitulo_padre(self, articulo: Dict, estructura: Dict) -> Optional[str]:
        """
        Encuentra el capítulo al que pertenece un artículo
        """
        for capitulo in reversed(estructura.get('capitulos', [])):
            if articulo['inicio'] > capitulo['inicio']:
                return capitulo['numero']
        return None
    
    def _guardar_documento(self, documento: Dict):
        """
        Guarda el documento y sus componentes en la base de datos
        """
        try:
            with self.engine.begin() as conn:
                # Insertar documento principal
                conn.execute(
                    text("""
                        INSERT INTO normas_legales (
                            id, categoria_principal, titulo, texto_norma,
                            palabras_clave, nombre_archivo, fecha_extraccion,
                            estado_vigencia, metadata_pdf
                        ) VALUES (
                            :id, :categoria, :titulo, :texto,
                            :palabras_clave, :nombre_archivo, :fecha_extraccion,
                            :estado, :metadata_pdf
                        )
                    """),
                    {
                        'id': documento['id'],
                        'categoria': documento['categoria_principal'],
                        'titulo': documento['titulo'],
                        'texto': documento['texto_norma'],
                        'palabras_clave': json.dumps(documento['palabras_clave']),
                        'nombre_archivo': documento['nombre_archivo'],
                        'fecha_extraccion': documento['fecha_extraccion'],
                        'estado': documento['estado'],
                        'metadata_pdf': json.dumps(documento['metadata_pdf'])
                    }
                )
                
                # Insertar artículos
                for articulo in documento.get('articulos', []):
                    conn.execute(
                        text("""
                            INSERT INTO articulos (
                                documento_id, numero, contenido,
                                titulo_padre, capitulo_padre
                            ) VALUES (
                                :doc_id, :numero, :contenido,
                                :titulo_padre, :capitulo_padre
                            )
                        """),
                        {
                            'doc_id': documento['id'],
                            'numero': articulo['numero'],
                            'contenido': articulo['contenido'],
                            'titulo_padre': articulo.get('titulo_padre'),
                            'capitulo_padre': articulo.get('capitulo_padre')
                        }
                    )
                    
            except Exception as e:
            logger.error(f"Error guardando documento: {str(e)}")
            raise

class TextStructureProcessor:
    def __init__(self):
        """Inicializa el procesador de estructura de texto"""
        # Patrones para identificar secciones
        self.patrones = {
            'titulo': r'TÍTULO\s+([IVX]+)[:\s-]+(.+?)(?=TÍTULO|\Z)',
            'capitulo': r'CAPÍTULO\s+([IVX]+)[:\s-]+(.+?)(?=CAPÍTULO|\Z)',
            'seccion': r'SECCIÓN\s+([IVX]+)[:\s-]+(.+?)(?=SECCIÓN|\Z)',
            'articulo': r'Artículo\s+(\d+)[°]?\.?[-:]?\s*(.+?)(?=Artículo|\Z)',
            'tipo_norma': r'(LEY|DECRETO SUPREMO|RESOLUCIÓN|ORDENANZA)\s+N°\s*(\d+[-\d\w]*)',
        }
        
        # Categorías y palabras clave
        self.categorias = {
            'CONSTITUCIONAL': ['constitución', 'constitucional', 'derechos fundamentales'],
            'ADMINISTRATIVO': ['administrativo', 'administración pública', 'procedimiento'],
            'PENAL': ['penal', 'delito', 'sanción', 'pena'],
            'CIVIL': ['civil', 'contratos', 'obligaciones'],
            'LABORAL': ['trabajo', 'laboral', 'trabajador'],
            'TRIBUTARIO': ['tributo', 'impuesto', 'contribución'],
            'AMBIENTAL': ['ambiental', 'ambiente', 'ecológico']
        }
        
        # Cargar modelo spaCy para procesamiento de lenguaje natural
        try:
            self.nlp = spacy.load('es_core_news_sm')
        except:
            logger.warning("Modelo spaCy no encontrado. Algunas funcionalidades estarán limitadas.")
            self.nlp = None

    def extract_structure(self, text: str) -> Dict:
        """Extrae la estructura jerárquica del texto"""
        estructura = {
            'titulos': [],
            'capitulos': [],
            'secciones': [],
            'articulos': [],
            'metadata': {}
        }
        
        # Extraer títulos
        for match in re.finditer(self.patrones['titulo'], text, re.DOTALL):
            estructura['titulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0)
            })
            
        # Extraer capítulos
        for match in re.finditer(self.patrones['capitulo'], text, re.DOTALL):
            estructura['capitulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0)
            })
            
        # Extraer secciones
        for match in re.finditer(self.patrones['seccion'], text, re.DOTALL):
            estructura['secciones'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0)
            })
            
        # Extraer artículos
        for match in re.finditer(self.patrones['articulo'], text, re.DOTALL):
            estructura['articulos'].append({
                'numero': match.group(1),
                'contenido': match.group(2).strip(),
                'texto': match.group(0)
            })
            
        return estructura

    def determine_category(self, text: str) -> Tuple[str, List[str]]:
        """Determina la categoría principal y palabras clave del texto"""
        text = text.lower()
        scores = {cat: 0 for cat in self.categorias}
        found_keywords = set()
        
        # Contar ocurrencias de palabras clave
        for categoria, keywords in self.categorias.items():
            for keyword in keywords:
                if keyword in text:
                    scores[categoria] += 1
                    found_keywords.add(keyword)
        
        # Determinar categoría principal
        if not any(scores.values()):
            return 'OTROS', list(found_keywords)
        
        categoria_principal = max(scores.items(), key=lambda x: x[1])[0]
        return categoria_principal, list(found_keywords)

    def extract_metadata(self, text: str) -> Dict:
        """Extrae metadatos del texto"""
        metadata = {
            'tipo_norma': None,
            'numero_norma': None,
            'fecha_extraccion': datetime.now().date().isoformat(),
            'estado_vigencia': 'VIGENTE',  # Por defecto
            'entidad_emisora': None,
            'ambito_aplicacion': 'NACIONAL'  # Por defecto
        }
        
        # Extraer tipo y número de norma
        match = re.search(self.patrones['tipo_norma'], text)
            if match:
            metadata['tipo_norma'] = match.group(1)
            metadata['numero_norma'] = match.group(2)

        return metadata

    def process_document(self, text: str, filename: str) -> Dict:
        """Procesa el documento completo y retorna la estructura"""
        # Extraer estructura jerárquica
        estructura = self.extract_structure(text)
        
        # Determinar categoría y palabras clave
        categoria, keywords = self.determine_category(text)
        
        # Extraer metadatos
        metadata = self.extract_metadata(text)
        
        # Crear documento estructurado
        documento = {
            'id': f"NORMA_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'categoria_principal': categoria,
            'subcategoria_1': None,  # Se puede implementar una lógica similar para subcategorías
            'subcategoria_2': None,
            'subcategoria_3': None,
            'titulo_numero': estructura['titulos'][0]['numero'] if estructura['titulos'] else None,
            'titulo_nombre': estructura['titulos'][0]['nombre'] if estructura['titulos'] else None,
            'capitulo_numero': estructura['capitulos'][0]['numero'] if estructura['capitulos'] else None,
            'capitulo_nombre': estructura['capitulos'][0]['nombre'] if estructura['capitulos'] else None,
            'seccion_numero': estructura['secciones'][0]['numero'] if estructura['secciones'] else None,
            'seccion_nombre': estructura['secciones'][0]['nombre'] if estructura['secciones'] else None,
            'articulo': estructura['articulos'][0]['numero'] if estructura['articulos'] else None,
            'titulo': estructura['titulos'][0]['nombre'] if estructura['titulos'] else None,
            'texto_norma': text,
            'palabras_clave': keywords,
            'nombre_archivo': filename,
            **metadata  # Incluir todos los metadatos extraídos
        }
        
        return documento

class PDFProcessor:
    def __init__(self):
        """Inicializa el procesador de PDFs"""
        self.min_text_length = 100  # Mínimo de caracteres para considerar texto válido
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae texto de un PDF usando múltiples métodos según sea necesario
        """
        texto_completo = ""
        
        try:
            # Primero intentar con PyMuPDF (más rápido y mejor detección de contenido)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Verificar si la página tiene texto
                if len(page.get_text().strip()) > self.min_text_length:
                    texto_completo += page.get_text() + "\n"
                else:
                    # Si no hay suficiente texto, intentar con OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    texto_ocr = pytesseract.image_to_string(img, lang='spa')
                    if texto_ocr.strip():
                        texto_completo += texto_ocr + "\n"
            
            # Si no se obtuvo suficiente texto, intentar con pdfplumber
            if len(texto_completo.strip()) < self.min_text_length:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        texto = page.extract_text()
                        if texto:
                            texto_completo += texto + "\n"
            
            return texto_completo.strip()
                
        except Exception as e:
            logger.error(f"Error procesando PDF {pdf_path}: {str(e)}")
            raise

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extrae metadatos del PDF
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Agregar información adicional
            metadata.update({
                'num_pages': len(doc),
                'has_text': any(len(page.get_text().strip()) > 0 for page in doc),
                'file_size': os.path.getsize(pdf_path),
                'extraction_date': datetime.now().isoformat()
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extrayendo metadatos de {pdf_path}: {str(e)}")
            return {}

class TextoLegalProcessor:
    def __init__(self):
        """Inicializa el procesador de texto legal"""
        # Patrones más flexibles usando regex para búsqueda aproximada
        self.patrones = {
            'titulo': regex.compile(r'T[IÍ]TULO\s*([IVX]+)[:\s-]*(.+?)(?=T[IÍ]TULO|\Z)', regex.IGNORECASE | regex.DOTALL),
            'capitulo': regex.compile(r'CAP[IÍ]TULO\s*([IVX]+)[:\s-]*(.+?)(?=CAP[IÍ]TULO|\Z)', regex.IGNORECASE | regex.DOTALL),
            'seccion': regex.compile(r'SECCI[OÓ]N\s*([IVX]+)[:\s-]*(.+?)(?=SECCI[OÓ]N|\Z)', regex.IGNORECASE | regex.DOTALL),
            'articulo': regex.compile(r'Art[ií]culo\s*(\d+)[°]?\.?[-:]?\s*(.+?)(?=Art[ií]culo|\Z)', regex.IGNORECASE | regex.DOTALL),
        }
        
        # Cargar modelo de NLP para español
        try:
            self.nlp = spacy.load('es_core_news_lg')  # Usar modelo más grande para mejor precisión
        except:
            logger.warning("Modelo spaCy grande no encontrado, usando modelo pequeño")
            self.nlp = spacy.load('es_core_news_sm')
        
        # Cargar modelo BERT para clasificación legal
        try:
            self.modelo_legal = AutoModelForSequenceClassification.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
            self.tokenizer = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo BERT: {str(e)}")
            self.modelo_legal = None
            self.tokenizer = None
    
    def extract_structure(self, text: str) -> Dict:
        """
        Extrae la estructura del texto legal con validación semántica
        """
        estructura = {
            'titulos': [],
            'capitulos': [],
            'secciones': [],
            'articulos': [],
            'metadata': {}
        }
        
        # Procesar el texto con spaCy para análisis lingüístico
        doc = self.nlp(text)
        
        # Extraer títulos con validación de contexto
        for match in self.patrones['titulo'].finditer(text):
            titulo = {
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            }
            
            # Validar contexto usando spaCy
            span = doc.char_span(match.start(), match.end())
            if span and any(token.pos_ == "NOUN" for token in span):
                estructura['titulos'].append(titulo)
        
        # Extraer capítulos y validar jerarquía
        for match in self.patrones['capitulo'].finditer(text):
            capitulo = {
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            }
            
            # Verificar que el capítulo pertenezca a un título
            titulo_padre = None
            for titulo in estructura['titulos']:
                if match.start() > titulo['inicio']:
                    titulo_padre = titulo
            
            if titulo_padre:
                capitulo['titulo_padre'] = titulo_padre['numero']
                estructura['capitulos'].append(capitulo)
        
        return estructura

class QualityValidator:
    def __init__(self, engine):
        """Inicializa el validador de calidad"""
        self.engine = engine
        self.elastic = Elasticsearch([processing_config.elasticsearch_url])
        
    def validate_document(self, documento: Dict) -> Tuple[bool, List[ErrorValidacion]]:
        """
        Valida un documento completo usando múltiples criterios
        """
        errores = []
        
        # Crear DataFrame para validación con Great Expectations
        df = pd.DataFrame([documento])
        ge_df = PandasDataset(df)
        
        # Validar campos requeridos
        ge_df.expect_column_values_to_not_be_null('id')
        ge_df.expect_column_values_to_not_be_null('texto_norma')
        ge_df.expect_column_values_to_not_be_null('titulo')
        
        # Validar longitud mínima del texto
        ge_df.expect_column_value_lengths_to_be_between(
            'texto_norma',
            min_value=100,
            mostly=1.0
        )
        
        # Validar formato de fechas
        if 'fecha_extraccion' in documento:
            ge_df.expect_column_values_to_match_strftime_format(
                'fecha_extraccion',
                strftime_format='%Y-%m-%d'
            )
        
        # Obtener resultados de validación
        validation_results = ge_df.validate()
        
        # Procesar resultados
        if not validation_results.success:
            for result in validation_results.results:
                if not result.success:
                    errores.append(ErrorValidacion(
                        tipo="ERROR_VALIDACION_DATOS",
                        descripcion=result.expectation_config.kwargs.get('description', 'Error de validación'),
                        severidad="ALTA",
                        contexto={"expectation": result.expectation_config.kwargs}
                    ))
        
        # Validar relaciones entre tablas
        if not self._validar_relaciones(documento):
            errores.append(ErrorValidacion(
                tipo="ERROR_RELACIONES",
                descripcion="Error en las relaciones entre tablas",
                severidad="ALTA",
                contexto={"documento_id": documento.get('id')}
            ))
        
        # Registrar errores en Elasticsearch para análisis
        if errores:
            self._registrar_errores(documento.get('id'), errores)
        
        return len(errores) == 0, errores
    
    def _validar_relaciones(self, documento: Dict) -> bool:
        """
        Valida las relaciones entre las diferentes tablas
        """
        try:
            with self.engine.connect() as conn:
                # Verificar que los artículos referencien al documento
                if documento.get('articulos'):
                    for articulo in documento['articulos']:
                        result = conn.execute(
                            text("""
                                SELECT COUNT(*) 
                                FROM articulos 
                                WHERE documento_id = :doc_id 
                                AND numero = :num
                            """),
                            {"doc_id": documento['id'], "num": articulo['numero']}
                        ).scalar()
                        
                        if result > 0:
                            return False  # Artículo duplicado
                
                return True
            
        except Exception as e:
            logger.error(f"Error validando relaciones: {str(e)}")
            return False
    
    def _registrar_errores(self, documento_id: str, errores: List[ErrorValidacion]):
        """
        Registra los errores en Elasticsearch para análisis
        """
        try:
            for error in errores:
                self.elastic.index(
                    index="errores_validacion",
                    document={
                        "documento_id": documento_id,
                        "tipo": error.tipo,
                        "descripcion": error.descripcion,
                        "severidad": error.severidad,
                        "contexto": error.contexto,
                        "timestamp": error.timestamp.isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error registrando en Elasticsearch: {str(e)}")

def main():
    """Función principal"""
    processor = NormaProcessor()
    # Aquí agregaremos la lógica para procesar documentos en batch
    
if __name__ == "__main__":
    main() 