from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import time
import os
import pandas as pd
from PyPDF2 import PdfReader
import json
from datetime import datetime
import logging
from io import BytesIO
import re
from transformers import pipeline
import spacy
from typing import Dict, List, Optional
from collections import defaultdict
import warnings
from tqdm import tqdm
import sys

# Suprimir advertencias específicas
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log'
)

class PeruanoScraper:
    def __init__(self):
        self.base_url = "https://diariooficial.elperuano.pe/normas/normasactualizadas"
        self.pdf_url = "https://diariooficial.elperuano.pe/Normas/obtenerDocumento"
        self.download_dir = os.path.abspath("pdfs")
        self.data_dir = "data"
        self.setup_directories()
        self.session = requests.Session()
        
        # Headers para simular un navegador
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Inicializar modelos de NLP con configuración optimizada
        try:
            print("Inicializando modelos de NLP...")
            self.nlp = spacy.load("es_core_news_lg")
            self.zero_shot = pipeline("zero-shot-classification", 
                                   model="facebook/bart-large-mnli", 
                                   device=-1)
            self.keyword_extractor = pipeline("zero-shot-classification", 
                                           model="facebook/bart-large-mnli", 
                                           device=-1)
            print("Modelos de NLP inicializados correctamente")
        except Exception as e:
            logging.error(f"Error inicializando modelos de NLP: {str(e)}")
            raise

        self.session_file = os.path.join(self.data_dir, 'scraping_session.json')
        self.current_session = self.load_session()

    def setup_directories(self):
        """Crear directorios necesarios si no existen"""
        try:
            for directory in [self.download_dir, self.data_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logging.info(f"Directorio creado: {directory}")
        except Exception as e:
            logging.error(f"Error creando directorios: {str(e)}")
            raise

    def extract_metadata(self, text: str) -> Dict:
        """Extraer metadatos usando IA más precisa"""
        try:
            # Limpiar y normalizar el texto
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Extraer título usando el primer párrafo significativo
            title = ""
            paragraphs = text.split('\n')
            for p in paragraphs:
                if len(p.strip()) > 20:  # Evitar líneas muy cortas
                    title = p.strip()
                    break
            
            # Extraer fecha usando regex mejorado
            date_patterns = [
                r'(\d{1,2}\s+de\s+[a-zA-Záéíóúñ]+\s+de\s+\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{4}-\d{2}-\d{2})'
            ]
            publication_date = None
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    publication_date = match.group(1)
                    break
            
            # Clasificar tipo de norma usando zero-shot
            norm_types = [
                "Decreto Supremo",
                "Resolución Ministerial",
                "Ley",
                "Ordenanza",
                "Resolución Directoral",
                "Decreto de Urgencia",
                "Decreto Legislativo",
                "Resolución Vice Ministerial",
                "Otro"
            ]
            
            # Usar el primer párrafo para clasificación
            classification_text = paragraphs[0] if paragraphs else ""
            classification = self.zero_shot(
                classification_text,
                candidate_labels=norm_types,
                multi_label=False
            )
            norm_type = classification['labels'][0]
            
            # Extraer entidades nombradas
            doc = self.nlp(text[:1000])  # Procesar solo los primeros 1000 caracteres
            entities = {
                'ORGANIZACION': [],
                'PERSONA': [],
                'LUGAR': [],
                'FECHA': []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Extraer referencias usando regex mejorado
            reference_patterns = [
                r'Referencia:\s*([^\n]+)',
                r'Ref\.\s*([^\n]+)',
                r'Referencia\s+N°\s*([^\n]+)'
            ]
            reference = None
            for pattern in reference_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    reference = match.group(1).strip()
                    break
            
            # Extraer artículos
            articles = []
            article_pattern = r'Artículo\s+(\d+)[°]?\s*[.-]\s*([^\n]+)'
            for match in re.finditer(article_pattern, text):
                articles.append({
                    'numero': match.group(1),
                    'contenido': match.group(2).strip()
                })
            
            # Extraer objetivos usando zero-shot
            objective_candidates = [
                "El objetivo de esta norma es",
                "Esta norma tiene por objeto",
                "La presente norma tiene como objetivo",
                "Objetivo de la norma"
            ]
            
            objectives = []
            for candidate in objective_candidates:
                if candidate.lower() in text.lower():
                    start_idx = text.lower().find(candidate.lower())
                    end_idx = text.find('.', start_idx)
                    if end_idx == -1:
                        end_idx = text.find('\n', start_idx)
                    if end_idx != -1:
                        objectives.append(text[start_idx:end_idx + 1].strip())
            
            # Extraer resumen usando el primer párrafo significativo
            summary = ""
            for p in paragraphs:
                if len(p.strip()) > 50 and not p.strip().startswith('Artículo'):
                    summary = p.strip()
                    break
            
            # Extraer palabras clave
            keywords = self.keyword_extractor(
                text[:500],  # Usar solo los primeros 500 caracteres
                candidate_labels=[
                    "Derecho", "Administración", "Economía", "Salud", 
                    "Educación", "Trabajo", "Ambiente", "Seguridad",
                    "Tributario", "Penal", "Civil", "Constitucional"
                ],
                multi_label=True
            )
            
            # Crear diccionario de metadatos estructurado
            metadata = {
                'Título': title,
                'Fecha de Publicación': publication_date,
                'Tipo de Norma': norm_type,
                'Referencia': reference,
                'Organizaciones': list(set(entities['ORGANIZACION'])),
                'Personas': list(set(entities['PERSONA'])),
                'Lugares': list(set(entities['LUGAR'])),
                'Fechas': list(set(entities['FECHA'])),
                'Artículos': articles,
                'Objetivos': objectives,
                'Resumen': summary,
                'Palabras Clave': keywords['labels'][:5],  # Top 5 keywords
                'Puntuaciones de Keywords': [round(score, 3) for score in keywords['scores'][:5]]
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error en extracción de metadatos: {str(e)}")
            return {
                'Título': '',
                'Fecha de Publicación': None,
                'Tipo de Norma': 'Desconocido',
                'Referencia': None,
                'Organizaciones': [],
                'Personas': [],
                'Lugares': [],
                'Fechas': [],
                'Artículos': [],
                'Objetivos': [],
                'Resumen': '',
                'Palabras Clave': [],
                'Puntuaciones de Keywords': []
            }

    def extract_title(self, text: str) -> str:
        """Extraer el título de la norma"""
        title_patterns = [
            r'(?:TÍTULO|TITULO)\s+[^\.]+',
            r'(?:QUE\s+[^\.]+)',
            r'(?:POR\s+CUANTO\s+[^\.]+)'
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def extract_norm_type_and_number(self, text: str) -> tuple:
        """Extraer tipo y número de norma"""
        norm_patterns = [
            r'(?:LEY|DECRETO SUPREMO|RESOLUCIÓN MINISTERIAL|ORDENANZA|REGLAMENTO)\s+N°\s*(\d+)',
            r'(?:LEY|DECRETO SUPREMO|RESOLUCIÓN MINISTERIAL|ORDENANZA|REGLAMENTO)\s+(\d+-\d+-\w+)'
        ]
        for pattern in norm_patterns:
            match = re.search(pattern, text)
            if match:
                norm_type = re.search(r'(?:LEY|DECRETO SUPREMO|RESOLUCIÓN MINISTERIAL|ORDENANZA|REGLAMENTO)', text).group(0)
                norm_number = match.group(1)
                return norm_type, norm_number
        return "Sin tipo", "Sin número"

    def extract_publication_date(self, text: str) -> str:
        """Extraer fecha de publicación"""
        date_patterns = [
            r'(?:publicado|publicada)\s+el\s+(\d{1,2}\s+de\s+[a-zA-Z]+\s+de\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return datetime.now().strftime("%Y-%m-%d")

    def extract_issuing_entity(self, text: str) -> str:
        """Extraer entidad emisora"""
        entity_patterns = [
            r'(?:EL PRESIDENTE DE LA REPÚBLICA|EL CONGRESO DE LA REPÚBLICA|EL MINISTRO|LA MINISTRA)',
            r'(?:EL PRESIDENTE|EL CONGRESO|EL MINISTRO|LA MINISTRA)\s+[^\.]+'
        ]
        for pattern in entity_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def extract_scope(self, text: str) -> str:
        """Extraer ámbito de aplicación"""
        scope_patterns = [
            r'(?:ÁMBITO DE APLICACIÓN|AMBITO DE APLICACION)[:\.]\s+[^\.]+',
            r'(?:APLICABLE|APLÍCASE)\s+A\s+[^\.]+'
        ]
        for pattern in scope_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "Nacional"

    def extract_keywords(self, text: str) -> str:
        """Extraer palabras clave"""
        try:
            # Usar el modelo de clasificación para identificar temas relevantes
            topics = [
                "Salud", "Educación", "Medio Ambiente", "Economía", "Seguridad",
                "Infraestructura", "Trabajo", "Agricultura", "Comercio", "Turismo"
            ]
            # Usar el modelo BART para clasificación
            classification = self.keyword_extractor(text, topics, multi_label=True)
            # Tomar las top 5 categorías con mayor confianza
            top_indices = sorted(range(len(classification['scores'])), 
                               key=lambda i: classification['scores'][i], 
                               reverse=True)[:5]
            return ", ".join([classification['labels'][i] for i in top_indices])
        except Exception as e:
            logging.error(f"Error extrayendo palabras clave: {str(e)}")
            return ""

    def classify_category(self, text: str) -> str:
        """Clasificar la categoría temática"""
        try:
            categories = [
                "Salud", "Educación", "Medio Ambiente", "Economía", "Seguridad",
                "Infraestructura", "Trabajo", "Agricultura", "Comercio", "Turismo"
            ]
            # Usar el modelo BART para clasificación
            classification = self.zero_shot(text, categories)
            return classification['labels'][0]
        except Exception as e:
            logging.error(f"Error clasificando categoría: {str(e)}")
            return "Sin categoría"

    def generate_link(self, norm_type: str, norm_number: str) -> str:
        """Generar enlace al documento"""
        return f"{self.base_url}/norma/{norm_type.lower().replace(' ', '-')}-{norm_number}"

    def generate_summary(self, text: str) -> str:
        """Generar resumen del texto"""
        try:
            # Usar spaCy para generar un resumen básico
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            if len(sentences) > 3:
                return " ".join(sentences[:3]) + "..."
            return text[:200] + "..."
        except Exception as e:
            logging.error(f"Error generando resumen: {str(e)}")
            return ""

    def download_pdf(self, id_norma):
        """Descargar PDF usando el ID de norma"""
        try:
            params = {'idNorma': str(id_norma)}
            response = self.session.get(self.pdf_url, params=params, headers=self.headers)
            
            if response.status_code == 200 and response.headers.get('content-type', '').lower() == 'application/pdf':
                pdf_path = os.path.join(self.download_dir, f'norma_{id_norma}.pdf')
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                return pdf_path
            else:
                logging.error(f"Error descargando PDF {id_norma}: Status {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error en la descarga del PDF {id_norma}: {str(e)}")
            return None

    def extract_pdf_text(self, pdf_path):
        """Extraer texto del PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logging.error(f"Error extrayendo texto del PDF: {str(e)}")
            return ""

    def process_single_norma(self, id_norma: int) -> Optional[Dict]:
        """Procesar una norma individual con feedback detallado"""
        try:
            print(f"\nProcesando norma ID: {id_norma}")
            start_time = time.time()
            
            # Descargar PDF
            print("Descargando PDF...")
            pdf_path = self.download_pdf(id_norma)
            
            if pdf_path and os.path.exists(pdf_path):
                # Extraer texto
                print("Extrayendo texto del PDF...")
                text = self.extract_pdf_text(pdf_path)
                
                if text.strip():
                    # Extraer metadatos
                    print("Extrayendo metadatos...")
                    metadata = self.extract_metadata(text)
                    metadata['ID'] = id_norma
                    
                    # Calcular tiempo
                    end_time = time.time()
                    processing_time = end_time - start_time
                    print(f"Tiempo de procesamiento: {processing_time:.2f} segundos")
                    
                    return metadata
                else:
                    print("PDF está vacío o no contiene texto")
            else:
                print("No se pudo descargar el PDF")
            
            return None
        except Exception as e:
            print(f"Error procesando norma {id_norma}: {str(e)}")
            return None

    def load_session(self):
        """Cargar o crear una nueva sesión"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self.create_new_session()
        return self.create_new_session()

    def create_new_session(self):
        """Crear una nueva sesión de scraping"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = {
            'session_id': timestamp,
            'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_processed_id': 0,
            'total_processed': 0,
            'data_file': 'normas_completas.csv',  # Archivo fijo para modo recomendado
            'json_file': 'normas_completas.json'   # Archivo fijo para modo recomendado
        }
        self.save_session(session)
        return session

    def save_session(self, session):
        """Guardar el estado de la sesión"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2)

    def process_normas_batch(self, start_id: int, end_id: int, batch_size: int = 10):
        """Procesar un lote de normas con control de sesión"""
        total_normas = end_id - start_id + 1
        total_batches = (total_normas + batch_size - 1) // batch_size
        
        print(f"\nProcesando {total_normas} normas en {total_batches} lotes de {batch_size}")
        print("=" * 50)
        
        all_data = []
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = start_id + (batch_num * batch_size)
            batch_end = min(batch_start + batch_size - 1, end_id)
            
            print(f"\nLote {batch_num + 1}/{total_batches} (IDs {batch_start}-{batch_end})")
            print("-" * 30)
            
            batch_data = []
            for id_norma in range(batch_start, batch_end + 1):
                metadata = self.process_single_norma(id_norma)
                if metadata:
                    batch_data.append(metadata)
            
            if batch_data:
                all_data.extend(batch_data)
                self.save_data(all_data)
                print(f"\nGuardados {len(batch_data)} documentos del lote actual")
                
                # Actualizar sesión
                self.current_session['last_processed_id'] = batch_end
                self.current_session['total_processed'] += len(batch_data)
                self.save_session(self.current_session)
            
            # Calcular tiempo estimado restante
            elapsed_time = time.time() - start_time
            avg_time_per_norma = elapsed_time / (batch_num + 1) / batch_size
            remaining_normas = total_normas - ((batch_num + 1) * batch_size)
            estimated_remaining_time = remaining_normas * avg_time_per_norma
            
            print(f"\nTiempo estimado restante: {estimated_remaining_time/60:.1f} minutos")
            
            # Preguntar si desea continuar después de cada lote de 10
            if batch_num < total_batches - 1:
                continue_batch = input("\n¿Desea continuar con el siguiente lote? (s/n): ")
                if continue_batch.lower() != 's':
                    print("\nProceso interrumpido por el usuario")
                    break
        
        total_time = time.time() - start_time
        print(f"\nProceso completado en {total_time/60:.1f} minutos")
        print(f"Total de normas procesadas exitosamente: {len(all_data)}")

    def save_data(self, data):
        """Guardar datos en formato JSON y CSV"""
        try:
            # Guardar JSON
            json_path = os.path.join(self.data_dir, self.current_session['json_file'])
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Datos guardados en JSON: {json_path}")
            
            # Guardar CSV
            csv_path = os.path.join(self.data_dir, self.current_session['data_file'])
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logging.info(f"Datos guardados en CSV: {csv_path}")
        except Exception as e:
            logging.error(f"Error al guardar datos: {str(e)}")

if __name__ == "__main__":
    try:
        print("Iniciando el proceso de scraping...")
        scraper = PeruanoScraper()
        
        # Menú principal
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Modo Recomendado (Descarga automática de PDFs)")
        print("2. Modo Personalizado (Configuración manual)")
        
        modo = input("\nSeleccione el modo (1/2): ")
        
        if modo == "1":
            # Modo Recomendado
            print("\n=== MODO RECOMENDADO ===")
            print("Este modo descargará automáticamente los PDFs en lotes de 10")
            print("Se le preguntará si desea continuar después de cada lote")
            print("Los datos se guardarán en archivos separados por fecha")
            
            confirm = input("\n¿Desea continuar con el modo recomendado? (s/n): ")
            if confirm.lower() == 's':
                # Verificar si hay una sesión anterior
                if scraper.current_session['last_processed_id'] > 0:
                    print(f"\nSe encontró una sesión anterior:")
                    print(f"ID último procesado: {scraper.current_session['last_processed_id']}")
                    print(f"Total procesado: {scraper.current_session['total_processed']}")
                    print(f"Archivo de datos: {scraper.current_session['data_file']}")
                    
                    continue_previous = input("\n¿Desea continuar con la sesión anterior? (s/n): ")
                    if continue_previous.lower() == 's':
                        start_id = scraper.current_session['last_processed_id'] + 1
                        end_id = start_id + 9  # Procesar los siguientes 10 PDFs
                    else:
                        # Crear nueva sesión
                        scraper.current_session = scraper.create_new_session()
                        start_id = 1
                        end_id = 10  # Empezar con los primeros 10 PDFs
                else:
                    # Primera ejecución
                    scraper.current_session = scraper.create_new_session()
                    start_id = 1
                    end_id = 10
                
                print(f"\nProcesando PDFs del {start_id} al {end_id}")
                scraper.process_normas_batch(start_id, end_id, batch_size=10)
            else:
                print("Proceso cancelado")
                
        elif modo == "2":
            # Modo Personalizado
            print("\n=== MODO PERSONALIZADO ===")
            print("En este modo puede configurar:")
            print("- ID inicial y final")
            print("- Tamaño del lote")
            print("- Nombre del archivo de salida")
            
            confirm = input("\n¿Desea continuar con el modo personalizado? (s/n): ")
            if confirm.lower() == 's':
                start_id = int(input("\nIngrese el ID inicial (1-1000): "))
                end_id = int(input("Ingrese el ID final (1-1000): "))
                batch_size = int(input("Ingrese el tamaño del lote (1-50): "))
                
                # Validar entradas
                start_id = max(1, min(start_id, 1000))
                end_id = max(1, min(end_id, 1000))
                batch_size = max(1, min(batch_size, 50))
                
                print("\nConfiguración personalizada:")
                print(f"Rango de IDs: {start_id} - {end_id}")
                print(f"Tamaño de lote: {batch_size}")
                print(f"Total de normas a procesar: {end_id - start_id + 1}")
                
                confirm = input("\n¿Desea continuar con esta configuración? (s/n): ")
                if confirm.lower() == 's':
                    scraper.process_normas_batch(start_id, end_id, batch_size)
                else:
                    print("Proceso cancelado")
            else:
                print("Proceso cancelado")
        else:
            print("Opción no válida")
            
    except Exception as e:
        logging.error(f"Error general: {str(e)}")
        print(f"Error: {str(e)}") 