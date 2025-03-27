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
            chrome_options.add_argument('--disable-gpu')  # Necesario en algunos sistemas
            chrome_options.add_argument('--remote-debugging-port=9222')  # Puerto para DevTools
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-software-rasterizer')
            chrome_options.add_argument('--window-size=1920,1080')  # Tamaño de ventana fijo
            chrome_options.add_argument('--start-maximized')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.binary_location = chromium_path
            
            # Configurar el servicio de ChromeDriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from webdriver_manager.core.os_manager import ChromeType
            
            service = Service(
                ChromeDriverManager(
                    chrome_type=ChromeType.CHROMIUM
                ).install()
            )
            
            # Crear directorio temporal para ChromeDriver
            import tempfile
            temp_dir = tempfile.mkdtemp()
            service.creation_flags = "--log-path={}/chromedriver.log".format(temp_dir)
            
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
            # Marcar checkbox de "Ver Títulos"
            ver_titulos = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='checkbox']"))
            )
            if not ver_titulos.is_selected():
                ver_titulos.click()
                time.sleep(2)  # Esperar a que se actualice la vista

            # Esperar a que la tabla se cargue
            self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tr"))
            )
            return True
        except Exception as e:
            logger.error(f"Error esperando resultados: {str(e)}")
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
            
            # Cargar página
            self.driver.get(self.base_url)
            
            # Esperar y verificar resultados
            if not self.wait_for_results():
                logger.error("No se pudieron cargar los resultados")
                return False

            documents = []
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]  # Ignorar encabezado
            
            for row in rows:
                doc_info = self.extract_document_info(row)
                if doc_info:
                    documents.append(doc_info)
                    
                    # Intentar descargar el PDF
                    if self.click_download(row):
                        logger.info(f"Descarga iniciada para: {doc_info['titulo']}")
                    
                    # Esperar entre descargas
                    time.sleep(2)

            # Guardar metadata
            if documents:
                self.save_to_s3({
                    'documentos': documents,
                    'total': len(documents),
                    'fecha_scraping': datetime.now().isoformat()
                }, 'metadata')
                
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