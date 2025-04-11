import boto3
import json
import os
from datetime import datetime
import logging
import sys
import argparse
import pandas as pd
from PyPDF2 import PdfReader
from text_structure import TextStructureProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        """Inicializa el procesador de PDFs"""
        self.s3_client = boto3.client('s3')
        self.text_processor = TextStructureProcessor()
        self.bucket_name = os.getenv('BUCKET_NAME')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae el texto completo de un archivo PDF
        """
        try:
            pdf_reader = PdfReader(pdf_path)
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF {pdf_path}: {str(e)}")
            raise

    def save_to_s3(self, data: dict, key: str) -> bool:
        """
        Guarda datos en S3
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(data, ensure_ascii=False),
                ContentType='application/json'
            )
            logger.info(f"Archivo guardado en S3: {key}")
            return True
        except Exception as e:
            logger.error(f"Error guardando en S3: {str(e)}")
            return False

    def save_to_csv(self, results: list, output_dir: str) -> bool:
        """
        Guarda los resultados en formato CSV
        """
        try:
            # Crear DataFrame con los datos estructurados
            data = []
            for result in results:
                row = {
                    'id': result.get('id', ''),
                    'categoria_principal': result.get('categoria_principal', ''),
                    'subcategoria_1': result.get('subcategoria_1', ''),
                    'subcategoria_2': result.get('subcategoria_2', ''),
                    'subcategoria_3': result.get('subcategoria_3', ''),
                    'titulo_numero': result.get('titulo_numero', ''),
                    'titulo_nombre': result.get('titulo_nombre', ''),
                    'capitulo_numero': result.get('capitulo_numero', ''),
                    'capitulo_nombre': result.get('capitulo_nombre', ''),
                    'seccion_numero': result.get('seccion_numero', ''),
                    'seccion_nombre': result.get('seccion_nombre', ''),
                    'articulo': result.get('articulo', ''),
                    'titulo': result.get('titulo', ''),
                    'texto_norma': result.get('texto_norma', ''),
                    'palabras_clave': '|'.join(result.get('palabras_clave', [])),
                    'tipo_norma': result.get('tipo_norma', ''),
                    'numero_norma': result.get('numero_norma', ''),
                    'entidad_emisora': result.get('entidad_emisora', ''),
                    'ambito_aplicacion': result.get('ambito_aplicacion', ''),
                    'estado_vigencia': result.get('estado_vigencia', ''),
                    'fecha_procesamiento': result.get('fecha_procesamiento', ''),
                    'archivo_original': result.get('archivo_original', ''),
                    'referencias_normativas': '|'.join(result.get('referencias_normativas', [])),
                    'modificaciones': '|'.join(result.get('modificaciones', []))
                }
                data.append(row)
            
            # Crear DataFrame
            df = pd.DataFrame(data)
            
            # Guardar CSV localmente
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"datos_entrenamiento_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"CSV guardado localmente: {csv_path}")
            
            # Guardar CSV en S3
            s3_key = f"processed_local/{csv_filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=df.to_csv(index=False).encode('utf-8'),
                ContentType='text/csv'
            )
            logger.info(f"CSV guardado en S3: {s3_key}")
            
            return True
                
        except Exception as e:
            logger.error(f"Error guardando CSV: {str(e)}")
            return False

    def process_pdfs(self, input_dir: str, output_dir: str) -> bool:
        """
        Procesa todos los PDFs en el directorio de entrada
        """
        try:
            # Verificar directorios
            if not os.path.exists(input_dir):
                logger.error(f"El directorio de entrada no existe: {input_dir}")
                return False
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Obtener lista de PDFs
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            if not pdf_files:
                logger.warning(f"No se encontraron archivos PDF en: {input_dir}")
                return False
            
            logger.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar")
            
            # Lista para almacenar resultados
            all_results = []
            
            # Procesar cada PDF
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(input_dir, pdf_file)
                    logger.info(f"Procesando: {pdf_file}")
                    
                    # Extraer texto
                    texto = self.extract_text_from_pdf(pdf_path)
                    if not texto:
                        logger.warning(f"No se pudo extraer texto de: {pdf_file}")
                        continue
                    
                    # Procesar documento
                    result = self.text_processor.process_document(texto)
                    
                    # Añadir información adicional
                    result['fecha_procesamiento'] = datetime.now().isoformat()
                    result['archivo_original'] = pdf_file
                    
                    # Guardar JSON individual
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    json_filename = f"{timestamp}_{os.path.splitext(pdf_file)[0]}.json"
                    json_path = os.path.join(output_dir, json_filename)
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"JSON guardado localmente: {json_path}")
                    
                    # Guardar en S3
                    s3_key = f"processed_local/{json_filename}"
                    self.save_to_s3(result, s3_key)
                    
                    # Añadir a resultados
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error procesando {pdf_file}: {str(e)}")
                    continue
            
            # Guardar CSV con todos los resultados
            if all_results:
                self.save_to_csv(all_results, output_dir)
                logger.info("Procesamiento completado exitosamente")
                return True
            else:
                logger.warning("No se procesó ningún documento correctamente")
                return False
                
        except Exception as e:
            logger.error(f"Error en el procesamiento: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Procesador de documentos PDF')
    parser.add_argument('--input-dir', required=True, help='Carpeta con PDFs a procesar')
    parser.add_argument('--output-dir', required=True, help='Carpeta para guardar resultados')
    args = parser.parse_args()
    
    processor = PDFProcessor()
    success = processor.process_pdfs(args.input_dir, args.output_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 