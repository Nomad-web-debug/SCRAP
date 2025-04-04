import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import boto3
import fitz  # PyMuPDF
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extrae texto de un PDF usando PyMuPDF
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)
        
        doc.close()
        return '\n'.join(text_parts)
        
    except Exception as e:
        logger.error(f"Error extrayendo texto de {pdf_path}: {str(e)}")
        return None

def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto extraído del PDF
    
    Realiza las siguientes operaciones:
    1. Normalización de espacios y saltos de línea
    2. Corrección de caracteres especiales
    3. Normalización de puntuación
    4. Corrección de formato de artículos
    5. Unificación de viñetas y numeración
    6. Eliminación de encabezados y pies de página repetitivos
    """
    try:
        # 1. Normalización básica
        text = text.strip()  # Eliminar espacios al inicio y final
        
        # 2. Normalizar saltos de línea
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # 3. Eliminar múltiples saltos de línea
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
            
        # 4. Normalizar espacios
        text = ' '.join(word for word in text.split(' ') if word)
        
        # 5. Corregir espacios en puntuación
        puntuacion = ['.', ',', ';', ':', ')', ']', '}']
        for p in puntuacion:
            text = text.replace(f' {p}', p)
        
        apertura = ['(', '[', '{']
        for a in apertura:
            text = text.replace(f'{a} ', a)
            
        # 6. Normalizar formato de artículos
        text = text.replace('Art.', 'Artículo')
        text = text.replace('ART.', 'Artículo')
        text = text.replace('art.', 'Artículo')
        
        # 7. Normalizar números romanos
        import re
        roman_pattern = r'\b[IVXLCDM]+\b'
        def normalize_roman(match):
            roman = match.group(0)
            if len(roman) <= 5:  # Evitar falsos positivos
                return roman
            return roman.title()
        text = re.sub(roman_pattern, normalize_roman, text)
        
        # 8. Normalizar viñetas y numeración
        text = text.replace('•', '-')
        text = text.replace('●', '-')
        text = text.replace('○', '-')
        text = text.replace('▪', '-')
        
        # 9. Normalizar comillas
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # 10. Eliminar caracteres de control y no imprimibles
        text = ''.join(char for char in text if char.isprintable())
        
        # 11. Normalizar guiones
        text = text.replace('–', '-').replace('—', '-')
        
        # 12. Corregir espacios en referencias
        text = re.sub(r'(?<=\d)\s+(?=[°º])', '', text)  # "1 °" -> "1°"
        text = re.sub(r'N\s*°', 'N°', text)  # "N °" -> "N°"
        
        # 13. Normalizar fechas
        text = re.sub(r'(\d{1,2})\s*°', r'\1°', text)  # "1 °" -> "1°"
        
        # 14. Eliminar encabezados y pies repetitivos
        lines = text.split('\n')
        cleaned_lines = []
        header_footer_threshold = 0.9  # 90% de similitud para considerar línea repetitiva
        
        from difflib import SequenceMatcher
        def lines_are_similar(line1, line2):
            return SequenceMatcher(None, line1, line2).ratio() > header_footer_threshold
        
        # Filtrar líneas repetitivas
        for i, line in enumerate(lines):
            is_repetitive = False
            if i > 0 and i < len(lines) - 1:
                # Verificar si la línea es similar a líneas cercanas
                prev_similar = lines_are_similar(line, lines[i-1])
                next_similar = i < len(lines)-1 and lines_are_similar(line, lines[i+1])
                is_repetitive = prev_similar or next_similar
            
            if not is_repetitive and line.strip():
                cleaned_lines.append(line)
        
        # 15. Reconstruir el texto
        text = '\n'.join(cleaned_lines)
        
        # 16. Normalizar espacios finales
        text = '\n'.join(line.strip() for line in text.split('\n'))
        while '  ' in text:
            text = text.replace('  ', ' ')
            
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error en la limpieza del texto: {str(e)}")
        return text  # Retornar texto original si hay error

def generate_prompt(text: str, filename: str) -> str:
    """
    Genera el prompt para Llama-2-70B
    """
    return f"""Analiza el siguiente texto legal del archivo {filename} y genera una estructura JSON detallada.
    
    Instrucciones específicas:
    1. Identifica el tipo de documento legal
    2. Extrae la estructura jerárquica completa
    3. Genera un ID único basado en el contenido
    4. Identifica artículos, capítulos y secciones
    5. Extrae referencias y citas
    6. Genera tags relevantes
    
    El resultado debe seguir este formato exacto:
    {{
        "id": "string",
        "documento": "string",
        "tipo_documento": "string",
        "estructura": {{
            "capitulos": [
                {{
                    "titulo": "string",
                    "articulos": [
                        {{
                            "numero": "string",
                            "contenido": "string",
                            "referencias": ["string"]
                        }}
                    ]
                }}
            ]
        }},
        "tags": ["string"]
    }}
    
    Texto a analizar:
    {text}
    
    Genera una respuesta JSON válida que siga exactamente el formato especificado.
    </response>"""

def invoke_llama(text: str, filename: str, endpoint_name: str) -> Dict:
    """
    Invoca el endpoint de SageMaker con Llama para procesar el texto
    """
    try:
        # Obtener región de AWS
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Crear cliente de SageMaker con la región específica
        runtime = boto3.client('sagemaker-runtime', region_name=region)
        sagemaker = boto3.client('sagemaker', region_name=region)
        
        # Obtener información del endpoint para determinar el modelo
        endpoint_info = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_info['EndpointConfigName']
        config_info = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = config_info['ProductionVariants'][0]['ModelName']
        
        # Generar prompt
        prompt = generate_prompt(text, filename)
        
        # Preparar payload
        payload = {
            "inputs": {
                "prompt": prompt,
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "model_name": model_name
            }
        }
        
        # Invocar endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Procesar respuesta
        response_body = json.loads(response['Body'].read().decode())
        return json.loads(response_body['generated_text'])
        
    except Exception as e:
        logger.error(f"Error invocando Llama en SageMaker: {str(e)}")
        raise

def validate_structure(data: Dict) -> bool:
    """
    Valida que la estructura tenga todos los campos requeridos
    """
    required_fields = [
        'id', 'documento', 'tipo_documento', 'rama_derecho',
        'articulo_numero', 'articulo_titulo', 'contenido'
    ]
    
    # Validar campos requeridos
    if not all(field in data for field in required_fields):
        return False
    
    # Validar formato de campos específicos
    if not isinstance(data.get('tags', []), list):
        return False
        
    if data.get('modificado') not in [True, False]:
        return False
        
    # Validar que el ID tenga formato correcto
    if not data.get('id', '').replace('-', '').replace('_', '').isalnum():
        return False
        
    return True

def clean_old_files(output_dir: str):
    """
    Limpia archivos antiguos dejando solo los más recientes
    """
    try:
        # Obtener todos los archivos
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        
        # Mantener solo el CSV más reciente
        if len(csv_files) > 1:
            csv_files.sort(reverse=True)  # Ordenar por nombre (que incluye timestamp)
            for old_csv in csv_files[1:]:  # Eliminar todos excepto el más reciente
                os.remove(os.path.join(output_dir, old_csv))
                logger.info(f"Eliminado CSV antiguo: {old_csv}")
        
        # Agrupar JSONs por ID de documento
        json_by_id = {}
        for json_file in json_files:
            # Extraer ID del documento del nombre del archivo
            doc_id = json_file.split('_')[0]  # Asumiendo formato "ID_timestamp.json"
            if doc_id not in json_by_id:
                json_by_id[doc_id] = []
            json_by_id[doc_id].append(json_file)
        
        # Mantener solo el JSON más reciente para cada documento
        for doc_id, files in json_by_id.items():
            if len(files) > 1:
                files.sort(reverse=True)  # Ordenar por nombre (que incluye timestamp)
                for old_json in files[1:]:  # Eliminar todos excepto el más reciente
                    os.remove(os.path.join(output_dir, old_json))
                    logger.info(f"Eliminado JSON antiguo: {old_json}")
    
    except Exception as e:
        logger.error(f"Error limpiando archivos antiguos: {str(e)}")

def save_results(results: List[Dict], output_dir: str):
    """
    Guarda los resultados en formato JSON y CSV, manteniendo solo los más recientes
    """
    try:
        # Limpiar archivos antiguos primero
        clean_old_files(output_dir)
        
        # Crear timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar cada resultado como JSON individual
        for result in results:
            # Usar ID del documento en el nombre del archivo
            doc_id = result['id'].replace('/', '_').replace('\\', '_')
            json_filename = f"{doc_id}_{timestamp}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Guardado JSON: {json_filename}")
        
        # Guardar todos los resultados en un solo CSV
        df = pd.DataFrame(results)
        
        # Convertir listas a strings para CSV
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
        
        # Guardar CSV
        csv_filename = f"resultados_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Guardado CSV: {csv_filename}")
        
        # Crear archivo de índice HTML
        create_index_html(output_dir, csv_filename, [f"{r['id']}_{timestamp}.json" for r in results])
        
        logger.info(f"Resultados guardados en {output_dir}")
        
    except Exception as e:
        logger.error(f"Error guardando resultados: {str(e)}")
        raise

def create_index_html(output_dir: str, csv_file: str, json_files: List[str]):
    """
    Crea un archivo HTML con enlaces a los resultados
    """
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resultados del Procesamiento</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                }}
                .file-list {{
                    border: 1px solid #ddd;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
                .file-link {{
                    display: block;
                    padding: 10px;
                    margin: 5px 0;
                    text-decoration: none;
                    color: #333;
                }}
                .file-link:hover {{
                    background-color: #f5f5f5;
                }}
                .csv-link {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 0;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <h1>Resultados del Procesamiento</h1>
            <div class="file-list">
                <h2>CSV Consolidado</h2>
                <a href="{csv_file}" class="csv-link">Descargar CSV</a>
                <p class="timestamp">Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="file-list">
                <h2>Archivos JSON por Documento</h2>
        """
        
        for json_file in sorted(json_files):
            html_content += f'        <a href="{json_file}" class="file-link">{json_file}</a>\n'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Guardar archivo HTML
        html_path = os.path.join(output_dir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Creado archivo de índice HTML: {html_path}")
        
    except Exception as e:
        logger.error(f"Error creando archivo HTML: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Procesa PDFs con Llama en SageMaker')
    parser.add_argument('--input-dir', required=True, help='Directorio con PDFs')
    parser.add_argument('--output-dir', required=True, help='Directorio para resultados')
    parser.add_argument('--endpoint', required=True, help='Nombre del endpoint de SageMaker')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lista para almacenar todos los resultados
    all_results = []
    
    # Procesar cada PDF en el directorio
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.pdf'):
            logger.info(f"Procesando {filename}...")
            
            try:
                # Extraer texto del PDF
                pdf_path = os.path.join(args.input_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                
                if text:
                    # Limpiar texto
                    text = clean_text(text)
                    
                    # Procesar con Llama en SageMaker
                    result = invoke_llama(text, filename, args.endpoint)
                    
                    # Validar estructura
                    if validate_structure(result):
                        # Añadir a resultados
                        all_results.append(result)
                        logger.info(f"Documento {filename} procesado exitosamente")
                    else:
                        logger.error(f"Estructura inválida para {filename}")
                else:
                    logger.error(f"No se pudo extraer texto de {filename}")
                    
            except Exception as e:
                logger.error(f"Error procesando {filename}: {str(e)}")
    
    # Guardar todos los resultados
    if all_results:
        save_results(all_results, args.output_dir)
        logger.info(f"Procesamiento completado. {len(all_results)} documentos procesados")
    else:
        logger.warning("No se procesó ningún documento correctamente")

if __name__ == '__main__':
    main() 