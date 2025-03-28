import os
import json
import time
import boto3
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
import glob
import PyPDF2
import re  # Agregando importación de re
from dotenv import load_dotenv
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormasActualizadasScraper:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('BUCKET_NAME')
        self.base_url = "https://diariooficial.elperuano.pe/normas/normasactualizadas"
        self.backup_url = "https://diariooficial.elperuano.pe/normas"
        
        # Configurar directorios
        self.download_dir = "/tmp/pdfs"
        self.screenshot_dir = "/tmp/screenshots"
        self.feedback_file = "categorization_feedback.json"
        self.processed_docs_file = "processed_documents.json"
        
        # Crear directorios necesarios
        for directory in [self.download_dir, self.screenshot_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Error limpiando archivo {file_path}: {e}")
            else:
                os.makedirs(directory, exist_ok=True)
        
        # Inicializar el driver
        self.setup_driver()
        
        # Reiniciar registro de documentos procesados
        self.processed_docs = {
            "documentos": {},
            "ultima_actualizacion": datetime.now().isoformat()
        }
        
        # Cargar retroalimentación existente
        self.feedback_data = self.load_feedback_data()
        
        # Contador para IDs únicos
        self.id_counter = int(datetime.now().timestamp())

    def load_feedback_data(self):
        """Cargar datos de retroalimentación de categorización"""
        try:
            # Intentar obtener del bucket S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"feedback/{self.feedback_file}"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except:
            # Si no existe, crear estructura inicial
            return {
                "categorias_no_identificadas": [],
                "sugerencias_mejora": {},
                "estadisticas": {
                    "total_documentos": 0,
                    "documentos_sin_categoria": 0,
                    "categorias_mas_comunes": {}
                }
            }

    def save_feedback_data(self):
        """Guardar datos de retroalimentación en S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"feedback/{self.feedback_file}",
                Body=json.dumps(self.feedback_data, indent=2, ensure_ascii=False)
            )
            logger.info("Datos de retroalimentación guardados exitosamente")
        except Exception as e:
            logger.error(f"Error guardando datos de retroalimentación: {str(e)}")

    def generate_unique_id(self, tipo_norma, nro_norma):
        """Genera un ID único para cada documento"""
        self.id_counter += 1
        base = f"{tipo_norma}_{nro_norma}".replace('/', '_').replace(' ', '_')
        return f"{base}_{self.id_counter}"

    def setup_driver(self):
        """Configura el driver de Selenium"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            
            # Configuraciones para descargas
            prefs = {
                'download.default_directory': self.download_dir,
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': True
            }
            chrome_options.add_experimental_option('prefs', prefs)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 20)
            
            logger.info("Driver de Selenium configurado correctamente")
            
        except Exception as e:
            logger.error(f"Error configurando el driver: {str(e)}")
            raise

    def wait_for_results(self):
        """Espera a que los resultados se carguen"""
        try:
            logger.info("Esperando a que la página cargue...")
            
            # Esperar a que el DOM esté completamente cargado
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Dar tiempo para que la página se cargue completamente
            time.sleep(5)
            
            # Buscar tabla de resultados
            table = None
            table_selectors = ["table", ".table", "#tablaResultados", "//table"]
            
            for selector in table_selectors:
                try:
                    if selector.startswith("//"):
                        table = self.wait.until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    else:
                        table = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    if table:
                        logger.info(f"Tabla encontrada usando selector: {selector}")
                        break
                except:
                    continue
            
            if not table:
                logger.error("No se pudo encontrar la tabla de resultados")
                return False
            
            # Verificar si hay filas en la tabla
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) > 1:
                logger.info(f"Se encontraron {len(rows)} filas en la tabla")
                return True
            else:
                logger.warning("La tabla está vacía")
                return False
                
        except Exception as e:
            logger.error(f"Error esperando resultados: {str(e)}")
            # Tomar screenshot para diagnóstico
            try:
                screenshot_path = os.path.join(self.screenshot_dir, "error_screenshot.png")
                self.driver.save_screenshot(screenshot_path)
                # Subir screenshot a S3 para diagnóstico
                self.s3_client.upload_file(
                    screenshot_path,
                    self.bucket_name,
                    f"diagnostics/screenshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_error.png"
                )
                logger.info(f"Screenshot guardado y subido a S3")
            except:
                logger.error("No se pudo guardar el screenshot")
            return False

    def categorize_document(self, titulo, materia, texto_completo):
        """Categoriza un documento basado en su título, materia y texto completo"""
        categorias = {
            'CONSTITUCIONAL': {
                'keywords': ['constitución', 'constitucional', 'derechos fundamentales', 'garantías', 'reforma'],
                'subcategorias': {
                    'DERECHOS_FUNDAMENTALES': ['derechos humanos', 'libertades', 'garantías constitucionales', 'dignidad', 'igualdad'],
                    'PODERES_ESTADO': ['ejecutivo', 'legislativo', 'judicial', 'organismos constitucionales', 'autonomos'],
                    'REFORMA_CONSTITUCIONAL': ['reforma', 'modificación constitucional', 'enmienda'],
                    'CONTROL_CONSTITUCIONAL': ['tribunal constitucional', 'inconstitucionalidad', 'precedente vinculante']
                }
            },
            'ADMINISTRATIVO': {
                'keywords': ['administrativo', 'público', 'estado', 'gestión', 'servicio civil', 'procedimiento'],
                'subcategorias': {
                    'PROCEDIMIENTOS': ['procedimiento administrativo', 'tupa', 'silencio administrativo', 'recursos administrativos'],
                    'FUNCION_PUBLICA': ['servidor público', 'funcionario', 'servicio civil', 'carrera pública', 'servir'],
                    'CONTRATACIONES': ['contratación', 'adquisición', 'licitación', 'obras públicas', 'proveedores'],
                    'SISTEMAS_ADMINISTRATIVOS': ['presupuesto', 'tesorería', 'contabilidad', 'control', 'inversión pública'],
                    'RESPONSABILIDAD_ADMINISTRATIVA': ['sanción', 'procedimiento sancionador', 'faltas administrativas']
                }
            },
            'LABORAL': {
                'keywords': ['trabajo', 'laboral', 'empleados', 'trabajadores', 'compensación', 'remuneración', 'sindical'],
                'subcategorias': {
                    'DERECHOS_LABORALES': ['jornada', 'descanso', 'vacaciones', 'beneficios sociales', 'gratificaciones', 'cts'],
                    'SEGURIDAD_SOCIAL': ['pensiones', 'seguridad social', 'prestaciones', 'jubilación', 'afp', 'onp', 'essalud'],
                    'RELACIONES_COLECTIVAS': ['sindicatos', 'negociación colectiva', 'huelga', 'derecho sindical', 'convenios'],
                    'SEGURIDAD_SALUD': ['seguridad', 'salud ocupacional', 'accidentes trabajo', 'enfermedades profesionales'],
                    'REGIMENES_ESPECIALES': ['cas', 'servir', 'microempresa', 'régimen', 'trabajadores públicos']
                }
            },
            'TRIBUTARIO_FINANCIERO': {
                'keywords': ['tributario', 'impuesto', 'fiscal', 'tributo', 'contribución', 'financiero', 'bancario'],
                'subcategorias': {
                    'IMPUESTOS': ['renta', 'igv', 'predial', 'alcabala', 'isc', 'tributación municipal'],
                    'ADUANAS': ['importación', 'exportación', 'aranceles', 'drawback', 'comercio exterior'],
                    'PROCEDIMIENTOS_TRIBUTARIOS': ['fiscalización', 'cobranza', 'devolución', 'tribunal fiscal', 'sunat'],
                    'SISTEMA_FINANCIERO': ['bancos', 'seguros', 'afp', 'mercado valores', 'sbs'],
                    'PREVENCION_LAVADO': ['lavado activos', 'financiamiento terrorismo', 'uif', 'compliance']
                }
            },
            'PENAL': {
                'keywords': ['penal', 'delito', 'criminal', 'sanción', 'pena', 'procesal penal'],
                'subcategorias': {
                    'DELITOS': ['tipos penales', 'corrupción', 'lavado', 'crimen organizado', 'delitos informáticos'],
                    'PROCESO_PENAL': ['investigación', 'juicio', 'prisión preventiva', 'prueba penal'],
                    'SISTEMA_PENITENCIARIO': ['cárcel', 'prisión', 'beneficios penitenciarios', 'inpe'],
                    'JUSTICIA_JUVENIL': ['menores', 'adolescentes infractores', 'medidas socioeducativas'],
                    'COMPLIANCE_PENAL': ['responsabilidad empresas', 'prevención delitos', 'programas cumplimiento']
                }
            },
            'CIVIL_COMERCIAL': {
                'keywords': ['civil', 'personas', 'familia', 'contratos', 'obligaciones', 'comercial', 'empresarial'],
                'subcategorias': {
                    'PERSONAS_FAMILIA': ['capacidad', 'estado civil', 'matrimonio', 'divorcio', 'filiación', 'alimentos'],
                    'REALES_REGISTRAL': ['propiedad', 'posesión', 'garantías reales', 'registros públicos'],
                    'CONTRATOS_OBLIGACIONES': ['contratos', 'obligaciones', 'responsabilidad civil', 'indemnización'],
                    'SOCIETARIO': ['sociedades', 'empresas', 'accionistas', 'gobierno corporativo'],
                    'MERCADO_COMPETENCIA': ['protección consumidor', 'competencia desleal', 'libre competencia', 'indecopi']
                }
            },
            'AMBIENTAL_RECURSOS': {
                'keywords': ['ambiental', 'ambiente', 'ecológico', 'recursos naturales', 'biodiversidad', 'energía'],
                'subcategorias': {
                    'RECURSOS_NATURALES': ['agua', 'forestal', 'minería', 'hidrocarburos', 'pesca', 'concesiones'],
                    'PROTECCION_AMBIENTAL': ['contaminación', 'residuos', 'emisiones', 'calidad ambiental', 'eia'],
                    'CONSERVACION': ['áreas protegidas', 'especies', 'ecosistemas', 'biodiversidad', 'patrimonio natural'],
                    'CAMBIO_CLIMATICO': ['clima', 'gases efecto invernadero', 'energía renovable', 'bonos carbono'],
                    'COMUNIDADES': ['pueblos indígenas', 'consulta previa', 'derechos ancestrales']
                }
            },
            'REGULACION_SECTORIAL': {
                'keywords': ['regulación', 'sector', 'servicios públicos', 'telecomunicaciones', 'transporte'],
                'subcategorias': {
                    'TELECOMUNICACIONES': ['telecomunicaciones', 'internet', 'radiodifusión', 'espectro', 'osiptel'],
                    'ENERGIA_MINERIA': ['electricidad', 'hidrocarburos', 'minería', 'osinergmin', 'concesiones'],
                    'TRANSPORTE': ['transporte', 'infraestructura', 'puertos', 'aeropuertos', 'ositran'],
                    'SANEAMIENTO': ['agua potable', 'saneamiento', 'sunass', 'eps'],
                    'SALUD': ['salud', 'medicamentos', 'establecimientos salud', 'susalud', 'digemid']
                }
            },
            'PROCESAL_JUSTICIA': {
                'keywords': ['procesal', 'judicial', 'jurisdiccional', 'arbitraje', 'justicia'],
                'subcategorias': {
                    'PROCESO_CIVIL': ['proceso civil', 'medidas cautelares', 'ejecución', 'prueba'],
                    'PROCESO_CONSTITUCIONAL': ['amparo', 'habeas corpus', 'habeas data', 'cumplimiento'],
                    'ARBITRAJE_MEDIACION': ['arbitraje', 'conciliación', 'mediación', 'marc'],
                    'JUSTICIA_DIGITAL': ['expediente digital', 'notificación electrónica', 'firma digital'],
                    'ORGANIZACION_JUDICIAL': ['poder judicial', 'ministerio público', 'tribunales', 'juzgados']
                }
            }
        }

        texto_completo = f"{titulo.lower()} {materia.lower()} {texto_completo}" if texto_completo else f"{titulo.lower()} {materia.lower()}"
        resultado = {
            'categorias': [],
            'subcategorias': [],
            'keywords_encontradas': [],
            'relevancia': {}
        }

        # Buscar categorías principales y calcular relevancia
        for categoria, info in categorias.items():
            keywords_encontradas = [kw for kw in info['keywords'] if kw in texto_completo]
            if keywords_encontradas:
                resultado['categorias'].append(categoria)
                resultado['keywords_encontradas'].extend(keywords_encontradas)
                
                # Calcular relevancia basada en número de keywords encontradas
                relevancia = len(keywords_encontradas) / len(info['keywords'])
                resultado['relevancia'][categoria] = round(relevancia, 2)
                
                # Buscar subcategorías
                for subcategoria, subkeywords in info['subcategorias'].items():
                    if any(kw in texto_completo for kw in subkeywords):
                        resultado['subcategorias'].append(subcategoria)

        # Si no se encontró ninguna categoría, asignar OTROS
        if not resultado['categorias']:
            resultado['categorias'] = ['OTROS']
            resultado['subcategorias'] = ['GENERAL']
            resultado['relevancia'] = {'OTROS': 1.0}

        return resultado

    def load_processed_docs(self):
        """Cargar registro de documentos procesados"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"control/{self.processed_docs_file}"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except:
            return {
                "documentos": {},
                "ultima_actualizacion": datetime.now().isoformat()
            }

    def save_processed_docs(self):
        """Guardar registro de documentos procesados"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"control/{self.processed_docs_file}",
                Body=json.dumps(self.processed_docs, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )
            logger.info("Registro de documentos procesados actualizado")
        except Exception as e:
            logger.error(f"Error guardando registro de documentos: {str(e)}")

    def check_document_exists(self, id_norma: str, titulo: str) -> bool:
        """Verifica si un documento ya existe y está procesado"""
        doc_id = f"{id_norma}_{self.normalize_title(titulo)}"
        return doc_id in self.processed_docs["documentos"]

    def normalize_title(self, titulo: str) -> str:
        """Normaliza el título para usar como parte del ID"""
        return re.sub(r'[^\w]', '_', titulo.lower())

    def register_processed_document(self, id_norma: str, titulo: str, metadata: dict):
        """Registra un documento como procesado"""
        doc_id = f"{id_norma}_{self.normalize_title(titulo)}"
        self.processed_docs["documentos"][doc_id] = {
            "fecha_procesamiento": datetime.now().isoformat(),
            "metadata": metadata
        }
        self.processed_docs["ultima_actualizacion"] = datetime.now().isoformat()

    def extract_document_info(self, row):
        """Extrae información de una fila de la tabla"""
        try:
            cells = row.find_elements(By.TAG_NAME, 'td')
            if len(cells) >= 4:
                titulo = cells[0].text.strip()
                nro_norma = cells[1].text.strip()
                materia = cells[2].text.strip()
                
                # Generar ID único
                tipo_norma = self.detect_tipo_norma(titulo)
                id_norma = self.generate_unique_id(tipo_norma, nro_norma)
                
                # Descargar PDF y subir a S3
                pdf_path = None
                if self.click_download(row):
                    logger.info(f"Descarga iniciada para documento {id_norma}")
                    pdf_path = self.wait_for_download()
                    if pdf_path:
                        s3_key = f"raw/{id_norma}.pdf"
                        self.s3_client.upload_file(pdf_path, self.bucket_name, s3_key)
                        logger.info(f"PDF subido a S3: {s3_key}")
                        os.remove(pdf_path)
                
                # Obtener texto completo del PDF
                texto_completo = None
                if pdf_path:
                    texto_completo = self.extract_pdf_text(pdf_path)
                
                # Categorización
                categorizacion = self.categorize_document(titulo, materia, texto_completo)
                
                # Crear metadata
                metadata = {
                    'id': id_norma,
                    'titulo': titulo,
                    'numero': nro_norma,
                    'materia': materia,
                    'tipo_norma': tipo_norma,
                    'categorias': categorizacion['categorias'],
                    'subcategorias': categorizacion['subcategorias'],
                    'texto_completo': texto_completo,
                    'pdf_s3_key': s3_key if pdf_path else None,
                    'fecha_scraping': datetime.now().isoformat(),
                    'metadata': {
                        'relevancia': categorizacion['relevancia'],
                        'keywords': categorizacion['keywords_encontradas']
                    }
                }
                
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error extrayendo información: {str(e)}")
            return None

    def extract_pdf_text(self, pdf_path):
        """Extrae el texto completo del PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Limpiar el texto
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF: {str(e)}")
            return None

    def detect_tipo_norma(self, titulo):
        """Detecta el tipo de norma basado en el título"""
        tipos = {
            'LEY': r'\bLEY\b',
            'DECRETO_SUPREMO': r'\bDECRETO\s+SUPREMO\b',
            'DECRETO_LEGISLATIVO': r'\bDECRETO\s+LEGISLATIVO\b',
            'RESOLUCION': r'\bRESOLUCIÓN\b|\bRESOLUCION\b',
            'DIRECTIVA': r'\bDIRECTIVA\b',
            'REGLAMENTO': r'\bREGLAMENTO\b'
        }
        
        for tipo, patron in tipos.items():
            if re.search(patron, titulo.upper()):
                return tipo
        return 'OTROS'

    def wait_for_download(self):
        """Espera y retorna la ruta del último archivo descargado"""
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            files = glob.glob(os.path.join(self.download_dir, '*.pdf'))
            if files:
                return max(files, key=os.path.getctime)
            time.sleep(1)
        return None

    def determinar_grupo(self, texto):
        """Determina el grupo al que pertenece el texto basado en su contenido"""
        palabras_clave = {
            'GRUPO_A': ['derecho constitucional', 'derechos fundamentales', 'garantías constitucionales'],
            'GRUPO_B': ['derecho administrativo', 'procedimiento administrativo', 'gestión pública'],
            'GRUPO_C': ['derecho penal', 'proceso penal', 'delitos']
        }
        
        texto = texto.lower()
        puntuaciones = {grupo: 0 for grupo in palabras_clave}
        
        for grupo, keywords in palabras_clave.items():
            for keyword in keywords:
                if keyword in texto:
                    puntuaciones[grupo] += 1
        
        if not any(puntuaciones.values()):
            return 'NO_CLASIFICADO'
        
        return max(puntuaciones.items(), key=lambda x: x[1])[0]

    def click_download(self, row):
        """Hace clic en el botón de descarga y espera a que se complete"""
        try:
            # Intentar diferentes selectores en orden
            selectors = [
                ('xpath', '//input[@type="button" and @value="Descargar"]'),
                ('css', 'input[type="button"][value="Descargar"]'),
                ('xpath', '//input[@value="Descargar"]')
            ]
            
            for selector_type, selector in selectors:
                try:
                    if selector_type == 'xpath':
                        download_button = row.find_element(By.XPATH, selector)
                    else:
                        download_button = row.find_element(By.CSS_SELECTOR, selector)
                    
                    if download_button and download_button.is_displayed():
                        # Hacer scroll al botón
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                        time.sleep(1)
                        
                        # Intentar click
                        download_button.click()
                        
                        logger.info(f"Botón de descarga encontrado usando selector: {selector}")
                        time.sleep(2)
                        return True
                except:
                    continue
            
            logger.error("No se pudo encontrar el botón de descarga")
            return False
            
        except Exception as e:
            logger.error(f"Error haciendo clic en descarga: {str(e)}")
            return False

    def scrape(self):
        """Proceso principal de scraping"""
        try:
            logger.info("Iniciando scraping de normas actualizadas...")
            
            # Intentar con la URL principal
            logger.info(f"Intentando acceder a: {self.base_url}")
            self.driver.get(self.base_url)
            
            # Si falla, intentar con la URL de respaldo
            if not self.wait_for_results():
                logger.warning("Fallback: intentando con URL alternativa...")
                self.driver.get(self.backup_url)
                if not self.wait_for_results():
                    logger.error("No se pudieron cargar los resultados en ninguna URL")
                    return 0
            
            # Obtener el HTML actual para diagnóstico
            page_source = self.driver.page_source
            logger.info(f"Longitud del HTML: {len(page_source)}")
            
            documents = []
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]
            
            if not rows:
                logger.error("No se encontraron filas en la tabla")
                return 0
                
            logger.info(f"Procesando {len(rows)} filas...")
            
            for idx, row in enumerate(rows, 1):
                doc_info = self.extract_document_info(row)
                if doc_info:
                    documents.append(doc_info)
                    logger.info(f"Documento {idx} procesado: {doc_info['titulo'][:50]}...")
                    time.sleep(2)

            # Guardar metadata con estructura completa
            total_docs = len(documents)
            if documents:
                metadata = {
                    'documentos': documents,
                    'total': total_docs,
                    'fecha_scraping': datetime.now().strftime('%Y-%m-%d'),
                    'url_origen': self.driver.current_url,
                    'estadisticas': {
                        'categorias_encontradas': list(set(doc['categorias'][0] for doc in documents if doc['categorias'])),
                        'distribucion_tipos': {}
                    }
                }
                
                # Calcular distribución de tipos
                for doc in documents:
                    if doc['tipo_norma']:
                        metadata['estadisticas']['distribucion_tipos'][doc['tipo_norma']] = \
                            metadata['estadisticas']['distribucion_tipos'].get(doc['tipo_norma'], 0) + 1
                
                # Guardar metadata en S3
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metadata_key = f"metadata/scraping_{timestamp}.json"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(metadata, ensure_ascii=False, indent=2),
                    ContentType='application/json'
                )
                logger.info(f"Metadata guardada en S3: {metadata_key}")

                # Guardar retroalimentación y registro de procesados
                self.save_feedback_data()
                self.save_processed_docs()

                return total_docs
            else:
                logger.error("No se encontraron documentos para procesar")
                return 0
                
        except Exception as e:
            logger.error(f"Error en el proceso de scraping: {str(e)}")
            return 0
        finally:
            self.driver.quit()

if __name__ == '__main__':
    try:
        scraper = NormasActualizadasScraper()
        total_docs = scraper.scrape()
        print(f"Total de documentos procesados: {total_docs}")
        if total_docs > 0:
            logger.info("Proceso completado exitosamente")
            exit(0)
        else:
            logger.error("No se encontraron documentos para procesar")
            exit(1)
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")
        exit(1) 