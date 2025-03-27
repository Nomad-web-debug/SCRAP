import os
import json
import time
import boto3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
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
            
            # Dar tiempo adicional para que la página se renderice completamente
            time.sleep(5)
            
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

    def extract_document_info(self, row):
        """Extrae información de una fila de la tabla"""
        try:
            cells = row.find_elements(By.TAG_NAME, 'td')
            if len(cells) >= 4:
                return {
                    'titulo': cells[0].text.strip(),
                    'nro_norma': cells[1].text.strip(),
                    'materia': cells[2].text.strip(),
                    'fecha_scraping': datetime.now().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error extrayendo información: {str(e)}")
            return None

    def click_download(self, row):
        """Hace clic en el botón de descarga y espera a que se complete"""
        try:
            download_button = row.find_element(By.CSS_SELECTOR, '.descargar')
            download_button.click()
            time.sleep(2)  # Dar tiempo para que inicie la descarga
            return True
        except Exception as e:
            logger.error(f"Error haciendo clic en descarga: {str(e)}")
            return False

    def save_to_s3(self, data, key_prefix):
        """Guarda datos en S3"""
        try:
            key = f"{key_prefix}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(data, ensure_ascii=False),
                ContentType='application/json'
            )
            logger.info(f"Datos guardados en {key}")
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
                    return False
            
            # Obtener el HTML actual para diagnóstico
            page_source = self.driver.page_source
            logger.info(f"Longitud del HTML: {len(page_source)}")
            
            documents = []
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]  # Ignorar encabezado
            
            if not rows:
                logger.error("No se encontraron filas en la tabla")
                return False
                
            logger.info(f"Procesando {len(rows)} filas...")
            
            for idx, row in enumerate(rows, 1):
                doc_info = self.extract_document_info(row)
                if doc_info:
                    documents.append(doc_info)
                    logger.info(f"Documento {idx} procesado: {doc_info['titulo'][:50]}...")
                    
                    # Intentar descargar el PDF
                    if self.click_download(row):
                        logger.info(f"Descarga iniciada para documento {idx}")
                    
                    # Esperar entre descargas
                    time.sleep(2)

            # Guardar metadata
            if documents:
                self.save_to_s3({
                    'documentos': documents,
                    'total': len(documents),
                    'fecha_scraping': datetime.now().isoformat(),
                    'url_origen': self.driver.current_url
                }, 'metadata')
                logger.info(f"Metadata guardada para {len(documents)} documentos")
            else:
                logger.error("No se encontraron documentos para procesar")
                return False
                
            logger.info(f"Scraping completado. {len(documents)} documentos procesados")
            return True
            
        except Exception as e:
            logger.error(f"Error en el proceso de scraping: {str(e)}")
            return False
        finally:
            self.driver.quit()

if __name__ == '__main__':
    scraper = NormasActualizadasScraper()
    scraper.scrape() 