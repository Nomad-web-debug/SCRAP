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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormasActualizadasScraper:
    def __init__(self):
        self.base_url = "https://spij.minjus.gob.pe/normas/normasactualizadas"
        self.backup_url = "https://diariooficial.elperuano.pe/Normas/normasactualizadas"
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('BUCKET_NAME')
        self.setup_driver()

    def setup_driver(self):
        """Configura el driver de Selenium"""
        try:
            # Verificar que Chromium está instalado
            import shutil
            chromium_path = shutil.which('chromium')
            if not chromium_path:
                chromium_path = shutil.which('chromium-browser')
            if not chromium_path:
                chromium_path = '/snap/bin/chromium'  # Ruta específica para Ubuntu con snap
            
            if not chromium_path or not os.path.exists(chromium_path):
                raise Exception(f"No se encontró el binario de Chromium en {chromium_path}")
            
            logger.info(f"Usando Chromium en: {chromium_path}")
            
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')  # Nueva sintaxis para modo headless
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-software-rasterizer')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--disable-setuid-sandbox')
            chrome_options.add_argument('--disable-infobars')
            chrome_options.add_argument('--ignore-certificate-errors')
            chrome_options.add_argument('--remote-debugging-pipe')  # Usar pipe en lugar de puerto
            chrome_options.binary_location = chromium_path
            
            # Configurar el servicio de ChromeDriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from webdriver_manager.core.os_manager import ChromeType
            
            # Crear directorio temporal para logs
            import tempfile
            temp_dir = tempfile.mkdtemp()
            log_path = f"{temp_dir}/chromedriver.log"
            
            service = Service(
                ChromeDriverManager(
                    chrome_type=ChromeType.CHROMIUM
                ).install(),
                log_path=log_path
            )
            
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, 20)
            logger.info("Driver de Selenium configurado correctamente")
            
        except Exception as e:
            logger.error(f"Error configurando el driver: {str(e)}")
            raise

    def wait_for_results(self):
        """Espera a que los resultados se carguen usando Selenium"""
        try:
            logger.info("Esperando a que la página cargue...")
            
            # Esperar a que el DOM esté completamente cargado
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Esperar a que no haya más solicitudes AJAX pendientes
            self.driver.execute_script("""
                window.ajaxComplete = false;
                var oldSend = XMLHttpRequest.prototype.send;
                XMLHttpRequest.prototype.send = function() {
                    window.ajaxComplete = false;
                    oldSend.apply(this, arguments);
                    this.addEventListener('loadend', function() {
                        window.ajaxComplete = true;
                    });
                };
            """)
            
            # Dar tiempo adicional para que la página se renderice completamente
            time.sleep(10)  # Aumentar el tiempo de espera
            
            # Esperar a que termine cualquier animación
            self.driver.execute_script("""
                var lastHeight = document.body.scrollHeight;
                var checkCount = 0;
                var interval = setInterval(function() {
                    var currentHeight = document.body.scrollHeight;
                    if (currentHeight === lastHeight || checkCount > 10) {
                        clearInterval(interval);
                        window.heightStabilized = true;
                    }
                    lastHeight = currentHeight;
                    checkCount++;
                }, 500);
            """)
            
            # Esperar a que la altura se estabilice
            self.wait.until(lambda d: d.execute_script("return window.heightStabilized === true"))
            
            logger.info("Buscando checkbox 'Ver Títulos'...")
            # Intentar diferentes selectores para el checkbox
            checkbox_selectors = [
                "input[type='checkbox']",
                "#chkVerTitulos",
                "input[name='verTitulos']",
                "//input[@type='checkbox']"
            ]
            
            ver_titulos = None
            for selector in checkbox_selectors:
                try:
                    if selector.startswith("//"):
                        ver_titulos = self.wait.until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    else:
                        ver_titulos = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    if ver_titulos:
                        logger.info(f"Checkbox encontrado usando selector: {selector}")
                        break
                except:
                    continue
            
            if not ver_titulos:
                logger.error("No se pudo encontrar el checkbox 'Ver Títulos'")
                # Intentar continuar sin el checkbox
                pass
            else:
                if not ver_titulos.is_selected():
                    ver_titulos.click()
                    logger.info("Checkbox 'Ver Títulos' marcado")
                    time.sleep(3)  # Esperar a que se actualice la vista
            
            logger.info("Buscando tabla de resultados...")
            # Intentar diferentes selectores para la tabla
            table_selectors = [
                "table",
                ".table",
                "#tablaResultados",
                "//table"
            ]
            
            table = None
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
            if len(rows) > 1:  # Al menos el encabezado y una fila de datos
                logger.info(f"Se encontraron {len(rows)} filas en la tabla")
                return True
            else:
                logger.warning("La tabla está vacía")
                return False
                
        except Exception as e:
            logger.error(f"Error esperando resultados: {str(e)}")
            # Tomar screenshot para diagnóstico
            try:
                screenshot_path = "/tmp/error_screenshot.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Screenshot guardado en: {screenshot_path}")
            except:
                logger.error("No se pudo guardar el screenshot")
            return False

    def categorize_document(self, titulo, materia):
        """Categoriza un documento basado en su título y materia"""
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

        texto_completo = f"{titulo.lower()} {materia.lower()}"
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

    def extract_document_info(self, row):
        """Extrae información de una fila de la tabla"""
        try:
            cells = row.find_elements(By.TAG_NAME, 'td')
            if len(cells) >= 4:
                titulo = cells[0].text.strip()
                nro_norma = cells[1].text.strip()
                materia = cells[2].text.strip()
                
                # Obtener texto completo si está disponible
                texto_completo = None
                try:
                    # Intentar obtener el texto completo del PDF
                    texto_completo = self.extract_pdf_text(row)
                except Exception as e:
                    logger.warning(f"No se pudo extraer el texto completo: {str(e)}")
                
                # Obtener categorización
                categorizacion = self.categorize_document(titulo, materia)
                
                # Generar ID único basado en tipo de norma y número
                tipo_norma = self.detect_tipo_norma(titulo)
                id_norma = self.generate_id(tipo_norma, nro_norma)
                
                # Extraer año
                año = None
                match = re.search(r'\b(19|20)\d{2}\b', nro_norma)
                if match:
                    año = int(match.group())
                
                # Estructura completa con todos los campos
                return {
                    # Campos de identificación
                    'id': id_norma,
                    'titulo': titulo,
                    'numero': nro_norma,
                    'materia': materia,
                    'año': año,
                    
                    # Campos de categorización jerárquica
                    'categoria_principal': categorizacion['categorias'][0] if categorizacion['categorias'] else 'OTROS',
                    'subcategoria_1': categorizacion['subcategorias'][0] if categorizacion['subcategorias'] else 'GENERAL',
                    'subcategoria_2': categorizacion['subcategorias'][1] if len(categorizacion['subcategorias']) > 1 else None,
                    'subcategoria_3': categorizacion['subcategorias'][2] if len(categorizacion['subcategorias']) > 2 else None,
                    
                    # Campos de contenido
                    'texto_completo': texto_completo,  # Contenido completo del documento
                    'texto_resumen': self.extract_resumen(titulo, materia),
                    'palabras_clave': categorizacion['keywords_encontradas'],
                    
                    # Campos de origen y fuente
                    'tipo_norma': tipo_norma,
                    'origen': 'El Peruano - Normas Actualizadas',
                    'url_origen': self.driver.current_url,
                    'nombre_archivo': f"{tipo_norma}_{nro_norma.replace('/', '_')}.pdf" if tipo_norma and nro_norma else None,
                    
                    # Campos de metadata adicional
                    'fecha_scraping': datetime.now().strftime('%Y-%m-%d'),
                    'relevancia_categorias': categorizacion['relevancia'],
                    
                    # Campo de texto enriquecido para IA
                    'texto_contexto': self.generate_texto_contexto(titulo, tipo_norma, año, materia, categorizacion)
                }
            return None
        except Exception as e:
            logger.error(f"Error extrayendo información: {str(e)}")
            return None

    def extract_pdf_text(self, row):
        """Extrae el texto completo del PDF"""
        try:
            # Primero intentamos descargar el PDF
            if not self.click_download(row):
                return None
            
            # Esperar a que se complete la descarga
            time.sleep(3)
            
            # Buscar el archivo PDF más reciente en la carpeta de descargas
            downloads_path = os.path.expanduser("~/Downloads")
            list_of_files = glob.glob(os.path.join(downloads_path, '*.pdf'))
            if not list_of_files:
                return None
                
            latest_file = max(list_of_files, key=os.path.getctime)
            
            # Extraer texto del PDF
            with open(latest_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Limpiar el texto
            text = text.strip()
            
            # Eliminar el archivo temporal
            os.remove(latest_file)
            
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

    def generate_id(self, tipo_norma, nro_norma):
        """Genera un ID único para la norma"""
        # Limpiar número de norma
        nro_limpio = re.sub(r'[^\w]', '', nro_norma) if nro_norma else ''
        # Generar ID único
        return f"{tipo_norma}_{nro_limpio}_{datetime.now().strftime('%Y%m')}"

    def extract_resumen(self, titulo, materia):
        """Extrae un resumen basado en título y materia"""
        return f"{titulo}. {materia}".strip()

    def generate_texto_contexto(self, titulo, tipo_norma, año, materia, categorizacion):
        """Genera texto contextual enriquecido para IA"""
        return f"""
            Título: {titulo}
            Tipo de Norma: {tipo_norma if tipo_norma else 'No especificado'}
            Año: {año if año else 'No especificado'}
            Materia: {materia}
            Categorías Principales: {', '.join(categorizacion['categorias'])}
            Subcategorías: {', '.join(categorizacion['subcategorias'])}
            Palabras Clave: {', '.join(categorizacion['keywords_encontradas'])}
        """.strip()

    def click_download(self, row):
        """Hace clic en el botón de descarga y espera a que se complete"""
        try:
            # Lista de selectores específicos para los botones de la página
            download_selectors = [
                'input[type="button"][value="Descargar"]',
                'input.btn-primary[value="Descargar"]',
                '.btn-primary[value="Descargar"]',
                '//input[@type="button" and @value="Descargar"]',
                '//input[@class="btn-primary" and @value="Descargar"]'
            ]
            
            # Intentar cada selector
            for selector in download_selectors:
                try:
                    if selector.startswith('//'):
                        download_button = row.find_element(By.XPATH, selector)
                    else:
                        download_button = row.find_element(By.CSS_SELECTOR, selector)
                        
                    if download_button and download_button.is_displayed():
                        # Intentar hacer scroll al botón
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                        time.sleep(1)  # Esperar a que termine el scroll
                        
                        # Intentar click con JavaScript si el click normal falla
                        try:
                            download_button.click()
                        except:
                            self.driver.execute_script("arguments[0].click();", download_button)
                            
                        logger.info(f"Botón de descarga encontrado usando selector: {selector}")
                        time.sleep(2)  # Esperar a que inicie la descarga
                        return True
                except:
                    continue
            
            # Si no funcionó con los selectores anteriores, intentar buscar por el texto del botón
            try:
                buttons = row.find_elements(By.TAG_NAME, "input")
                for button in buttons:
                    if button.get_attribute("value") == "Descargar":
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(1)
                        try:
                            button.click()
                        except:
                            self.driver.execute_script("arguments[0].click();", button)
                        logger.info("Botón de descarga encontrado por texto")
                        time.sleep(2)
                        return True
            except:
                pass
                    
            # Si llegamos aquí, no se encontró el botón
            logger.error("No se pudo encontrar el botón de descarga")
            
            # Tomar screenshot y HTML para diagnóstico
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"/tmp/error_download_{timestamp}.png"
                html_path = f"/tmp/error_download_{timestamp}.html"
                
                self.driver.save_screenshot(screenshot_path)
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(row.get_attribute('outerHTML'))
                    
                logger.info(f"Screenshot guardado en: {screenshot_path}")
                logger.info(f"HTML guardado en: {html_path}")
            except:
                logger.error("No se pudo guardar el diagnóstico")
                
            return False
            
        except Exception as e:
            logger.error(f"Error haciendo clic en descarga: {str(e)}")
            return False

    def save_to_s3(self, data, key_prefix):
        """Guarda datos en S3 y genera un enlace público temporal"""
        try:
            # Generar nombre del archivo con timestamp
            key = f"{key_prefix}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Guardar en formato JSON legible
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(data, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            
            # Generar URL firmada para descarga (válida por 7 días)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=7*24*3600  # 7 días en segundos
            )
            
            logger.info(f"Datos guardados en S3: {key}")
            logger.info(f"Enlace de descarga (válido por 7 días): {url}")
            
            # Guardar el enlace en un archivo local para referencia
            with open('ultimo_enlace.txt', 'w') as f:
                f.write(f"Último enlace de descarga: {url}\n")
                f.write(f"Fecha generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Válido hasta: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error guardando en S3: {str(e)}")
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
                    
                    if self.click_download(row):
                        logger.info(f"Descarga iniciada para documento {idx}")
                    
                    time.sleep(2)

            # Guardar metadata con estructura completa
            total_docs = len(documents)
            if documents:
                # Recolectar todas las categorías y subcategorías
                todas_categorias = []
                todas_subcategorias = []
                for doc in documents:
                    if doc['categoria_principal'] != 'OTROS':
                        todas_categorias.append(doc['categoria_principal'])
                    if doc['subcategoria_1'] != 'GENERAL':
                        todas_subcategorias.append(doc['subcategoria_1'])
                    if doc['subcategoria_2']:
                        todas_subcategorias.append(doc['subcategoria_2'])
                    if doc['subcategoria_3']:
                        todas_subcategorias.append(doc['subcategoria_3'])

                metadata = {
                    'documentos': documents,
                    'total': total_docs,
                    'fecha_scraping': datetime.now().strftime('%Y-%m-%d'),
                    'url_origen': self.driver.current_url,
                    'estadisticas': {
                        'categorias_encontradas': list(set(todas_categorias)),
                        'subcategorias_encontradas': list(set(todas_subcategorias)),
                        'distribucion_años': {},
                        'distribucion_tipos': {}
                    }
                }
                
                # Calcular distribución de años y tipos
                for doc in documents:
                    if doc['año']:
                        metadata['estadisticas']['distribucion_años'][str(doc['año'])] = \
                            metadata['estadisticas']['distribucion_años'].get(str(doc['año']), 0) + 1
                    if doc['tipo_norma']:
                        metadata['estadisticas']['distribucion_tipos'][doc['tipo_norma']] = \
                            metadata['estadisticas']['distribucion_tipos'].get(doc['tipo_norma'], 0) + 1
                
                self.save_to_s3(metadata, 'metadata')
                logger.info(f"Metadata guardada para {total_docs} documentos")
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
        if total_docs == 0:
            exit(1)  # Salir con error si no se procesaron documentos
        exit(0)  # Salir exitosamente si se procesaron documentos
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")
        exit(1) 