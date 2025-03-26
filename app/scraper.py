import os
import logging
import logging.config
import asyncio
import aiohttp
import boto3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from config import scraping_config, LOGGING_CONFIG

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class BanxicoScraper:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=scraping_config.aws_region)
        self.session = None
        self.base_url = scraping_config.base_url
        os.makedirs(scraping_config.download_path, exist_ok=True)

    async def init_session(self):
        """Inicializar sesión HTTP asíncrona"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Cerrar sesión HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_pdf_links(self) -> List[Dict[str, str]]:
        """Obtener enlaces a PDFs de informes trimestrales"""
        try:
            async with self.session.get(self.base_url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                pdf_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf') and 'trimestral' in href.lower():
                        full_url = urljoin(self.base_url, href)
                        name = os.path.basename(href)
                        pdf_links.append({
                            'url': full_url,
                            'name': name
                        })
                
                logger.info(f"Encontrados {len(pdf_links)} PDFs para descargar")
                return pdf_links
        except Exception as e:
            logger.error(f"Error obteniendo enlaces de PDFs: {str(e)}")
            return []

    async def download_pdf(self, pdf_info: Dict[str, str]) -> bool:
        """Descargar un PDF y subirlo a S3"""
        url = pdf_info['url']
        name = pdf_info['name']
        local_path = os.path.join(scraping_config.download_path, name)

        try:
            # Descargar PDF
            async with self.session.get(url) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(scraping_config.chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

            # Subir a S3
            self.s3_client.upload_file(
                local_path,
                scraping_config.s3_bucket,
                f"raw/{name}"
            )
            
            logger.info(f"PDF descargado y subido exitosamente: {name}")
            os.remove(local_path)  # Limpiar archivo local
            return True

        except Exception as e:
            logger.error(f"Error procesando {name}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False

    async def process_pdfs(self):
        """Procesar todos los PDFs encontrados"""
        await self.init_session()
        try:
            pdf_links = await self.get_pdf_links()
            if not pdf_links:
                logger.error("No se encontraron PDFs para procesar")
                return

            # Procesar PDFs en paralelo
            tasks = []
            for pdf_info in pdf_links:
                task = asyncio.create_task(self.download_pdf(pdf_info))
                tasks.append(task)
                
                if len(tasks) >= scraping_config.max_concurrent_downloads:
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

        finally:
            await self.close_session()

class PeruanoScraper:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.session = None
        self.base_url = "https://diariooficial.elperuano.pe/Normas/obtenerDocumento"
        self.max_normas = 130
        os.makedirs(scraping_config.download_path, exist_ok=True)

    async def init_session(self):
        """Inicializar sesión HTTP asíncrona"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Cerrar sesión HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_pdf_urls(self) -> List[Dict[str, str]]:
        """Generar URLs de PDFs a descargar"""
        pdf_urls = []
        for i in range(1, self.max_normas + 1):
            url = f"{self.base_url}?idNorma={i}"
            pdf_urls.append({
                'url': url,
                'id': str(i),
                'name': f'norma_{i}.pdf'
            })
        return pdf_urls

    async def download_pdf(self, pdf_info: Dict[str, str]) -> bool:
        """Descargar un PDF y subirlo a S3"""
        url = pdf_info['url']
        name = pdf_info['name']
        norma_id = pdf_info['id']
        local_path = os.path.join(scraping_config.download_path, name)

        try:
            # Descargar PDF
            async with self.session.get(url) as response:
                if response.status == 200 and response.headers.get('content-type', '').startswith('application/pdf'):
                    with open(local_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(scraping_config.chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

                    # Subir a S3
                    self.s3_client.upload_file(
                        local_path,
                        scraping_config.s3_bucket,
                        f"normas/{name}"
                    )
                    
                    logger.info(f"Norma {norma_id} descargada y subida exitosamente")
                    os.remove(local_path)
                    return True
                else:
                    logger.warning(f"No se pudo descargar la norma {norma_id}: Status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Error procesando norma {norma_id}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False

    async def process_pdfs(self):
        """Procesar todos los PDFs"""
        await self.init_session()
        try:
            pdf_urls = await self.get_pdf_urls()
            
            # Procesar PDFs en paralelo
            tasks = []
            for pdf_info in pdf_urls:
                task = asyncio.create_task(self.download_pdf(pdf_info))
                tasks.append(task)
                
                if len(tasks) >= scraping_config.max_concurrent_downloads:
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

        finally:
            await self.close_session()

async def main():
    """Función principal"""
    scraper = BanxicoScraper()
    await scraper.process_pdfs()

    scraper = PeruanoScraper()
    await scraper.process_pdfs()

if __name__ == "__main__":
    asyncio.run(main()) 