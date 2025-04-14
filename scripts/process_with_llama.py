import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import boto3
import fitz  # PyMuPDF
import pandas as pd
import glob
import re

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
    Genera el prompt para el modelo Llama-2
    """
    return f"""Analiza el siguiente texto legal. Tu tarea es extraer e interpretar información específica manteniendo la precisión del texto original donde sea necesario.

IMPORTANTE - Reglas de extracción:
1. CAMPOS QUE DEBEN SER EXACTOS (copiar tal cual del texto):
   - Número de artículos
   - Títulos de capítulos/secciones
   - Contenido de artículos
   - Referencias y citas textuales
   - Nombres de leyes y decretos
   - Fechas

2. CAMPOS QUE REQUIEREN TU INTERPRETACIÓN:
   - Tipo de documento (basado en su estructura y contenido)
   - Tags relevantes
   - Clasificación de la rama del derecho
   - Resumen o descripción general

3. REGLAS DE PROCESAMIENTO:
   - NO inventes información que no esté en el texto
   - NO modifiques el texto de los artículos
   - NO agregues interpretaciones en campos que deben ser exactos
   - SI el texto no contiene alguna información requerida, usa null o lista vacía según corresponda

El resultado debe seguir este formato JSON:
{{
    "id": "string (generado del nombre del archivo)",
    "documento": "string (nombre exacto del documento)",
    "tipo_documento": "string (tu interpretación)",
    "estructura": {{
        "capitulos": [
            {{
                "titulo": "string (exacto del texto)",
                "articulos": [
                    {{
                        "numero": "string (exacto)",
                        "contenido": "string (exacto)",
                        "referencias": ["string (exactas)"]
                    }}
                ]
            }}
        ]
    }},
    "tags": ["string (tu interpretación)"]
}}

Texto a analizar:
{text}

Genera una respuesta JSON válida que siga exactamente el formato especificado, manteniendo la precisión donde se requiere.
</response>"""

def get_active_endpoint() -> Tuple[str, str]:
    """
    Obtiene el endpoint activo y el modelo desde el archivo de estado
    
    Returns:
        Tuple[str, str]: (nombre del endpoint, versión del modelo)
    """
    try:
        with open('data/endpoint_state.json', 'r') as f:
            estado = json.load(f)
            return estado['endpoint_name'], estado['modelo']
    except Exception as e:
        logger.error(f"Error leyendo estado del endpoint: {str(e)}")
        return None, None

def invoke_llama_model(text: str, endpoint_name: str, model_name: str, pdf_count: int) -> Optional[str]:
    """
    Invoca el modelo Llama 2 en SageMaker para procesar el texto
    
    Args:
        text (str): Texto a procesar
        endpoint_name (str): Nombre del endpoint de SageMaker
        model_name (str): Nombre del modelo a usar (ej: '7b', '13b', '70b')
        pdf_count (int): Número del PDF actual para control de logs
        
    Returns:
        Optional[str]: Texto generado por el modelo o None si hay error
    """
    try:
        # Obtener el endpoint activo actual
        active_endpoint, active_model = get_active_endpoint()
        if not active_endpoint or not active_model:
            logger.error("No se encontró un endpoint activo o modelo en el archivo de estado")
            return None
            
        region = os.environ.get('AWS_REGION', 'us-east-1')
        if pdf_count <= 15:
            logger.info(f"Iniciando invocación del modelo en región {region}")
            logger.info(f"Endpoint: {active_endpoint}")
            logger.info(f"Modelo: llama-2-{active_model}-chat")
        
        sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        # Generar el prompt
        prompt = generate_prompt(text, "")
        
        # Formato del payload para containers oficiales de SageMaker JumpStart
        payload = {
            "inputs": [[
                {"role": "system", "content": "Eres un asistente legal que analiza documentos legales."},
                {"role": "user", "content": prompt}
            ]],
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.2
            }
        }
        
        if pdf_count <= 15:
            logger.info("=== Detalles del Payload ===")
            logger.info(f"Payload completo: {json.dumps(payload, indent=2)}")
            logger.info(f"Configuración de parámetros: {json.dumps(payload['parameters'], indent=2)}")
            logger.info(f"Model name en CustomAttributes: llama-2-{active_model}-chat")
            logger.info("=== Fin Detalles del Payload ===")
        
        # Invocar endpoint con CustomAttributes para el model_name
        if pdf_count <= 15:
            logger.info(f"Invocando endpoint con payload de {len(json.dumps(payload))} bytes")
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=active_endpoint,
            ContentType='application/json',
            Body=json.dumps(payload),
            CustomAttributes=f"model_name=llama-2-{active_model}-chat"  # Usar active_model del archivo de estado
        )
        
        if pdf_count <= 15:
            logger.info("=== Detalles de la Respuesta ===")
            logger.info(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
            logger.info(f"Headers: {response['ResponseMetadata']['HTTPHeaders']}")
        
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        
        if pdf_count <= 15:
            logger.info(f"Response Body: {json.dumps(response_body, indent=2)}")
            logger.info("=== Fin Detalles de la Respuesta ===")
        
        if isinstance(response_body, list) and len(response_body) > 0:
            generated_text = response_body[0].get('generated_text', '')
            if pdf_count <= 15:
                logger.info(f"Texto generado extraído de lista de respuestas: {len(generated_text)} caracteres")
        elif isinstance(response_body, dict):
            generated_text = response_body.get('generated_text', '')
            if pdf_count <= 15:
                logger.info(f"Texto generado extraído de diccionario de respuesta: {len(generated_text)} caracteres")
        else:
            if pdf_count <= 15:
                logger.error(f"Formato de respuesta inesperado: {type(response_body)}")
                logger.error(f"Contenido de la respuesta: {response_body}")
            return None
            
        return generated_text
            
    except Exception as e:
        if pdf_count <= 15:
            logger.error("=== Error Detallado ===")
            logger.error(f"Tipo de error: {type(e).__name__}")
            logger.error(f"Mensaje de error: {str(e)}")
            logger.error(f"Endpoint: {active_endpoint}")
            logger.error(f"Modelo: llama-2-{active_model}-chat")
            logger.error(f"Región: {region}")
            logger.error("=== Fin Error Detallado ===")
        return None

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

def dividir_texto_por_secciones(text: str) -> List[str]:
    """
    Divide el texto en secciones basadas en patrones comunes de documentos legales.
    """
    try:
        # Patrones para identificar secciones
        patrones = [
            r'CAP[ÍI]TULO\s+[IVXLCDM]+',  # CAPÍTULO I, CAPITULO II, etc.
            r'T[ÍI]TULO\s+[IVXLCDM]+',    # TÍTULO I, TITULO II, etc.
            r'Art[íi]culo\s+\d+',          # Artículo 1, Articulo 2, etc.
            r'Secci[óo]n\s+\d+'            # Sección 1, Seccion 2, etc.
        ]
        
        # Unir todos los patrones
        patron = '|'.join(patrones)
        
        # Encontrar todas las coincidencias
        matches = list(re.finditer(patron, text))
        
        if not matches:
            # Si no hay secciones claras, dividir por longitud
            max_length = 3000  # Aproximadamente 3000 caracteres
            return [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        # Dividir el texto en secciones
        secciones = []
        start = 0
        
        for match in matches:
            # Agregar la sección anterior
            if match.start() > start:
                seccion = text[start:match.start()].strip()
                if seccion:
                    secciones.append(seccion)
            start = match.start()
        
        # Agregar la última sección
        if start < len(text):
            seccion = text[start:].strip()
            if seccion:
                secciones.append(seccion)
        
        return secciones
        
    except Exception as e:
        logger.error(f"Error dividiendo texto en secciones: {str(e)}")
        # Si hay error, dividir por longitud
        max_length = 3000
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def procesar_texto_largo(text: str, endpoint_name: str, model_name: str, pdf_count: int) -> Optional[Dict]:
    """
    Procesa un texto largo dividiéndolo en secciones y combinando los resultados.
    """
    try:
        # Obtener el endpoint activo actual
        active_endpoint, active_model = get_active_endpoint()
        if not active_endpoint:
            logger.error("No se encontró un endpoint activo en el archivo de estado")
            return None
            
        # Dividir el texto en secciones
        secciones = dividir_texto_por_secciones(text)
        if pdf_count <= 15:
            logger.info(f"Texto dividido en {len(secciones)} secciones")
        
        resultados = []
        
        # Procesar cada sección
        for i, seccion in enumerate(secciones, 1):
            if pdf_count <= 15:
                logger.info(f"Procesando sección {i}/{len(secciones)}")
            
            # Invocar modelo para la sección usando el endpoint activo
            resultado = invoke_llama_model(seccion, active_endpoint, model_name, pdf_count)
            
            if resultado is None:
                if pdf_count <= 15:
                    logger.error(f"Error procesando sección {i}")
                continue
                
            # Validar estructura
            if not validate_structure(resultado):
                if pdf_count <= 15:
                    logger.error(f"Estructura inválida en sección {i}")
                continue
                
            resultados.append(resultado)
        
        if not resultados:
            if pdf_count <= 15:
                logger.error("No se pudo procesar ninguna sección correctamente")
            return None
            
        # Combinar resultados
        resultado_final = combinar_resultados(resultados)
        return resultado_final
        
    except Exception as e:
        if pdf_count <= 15:
            logger.error(f"Error procesando texto largo: {str(e)}")
        return None

def combinar_resultados(resultados: List[Dict]) -> Dict:
    """
    Combina los resultados de múltiples secciones en un solo resultado.
    """
    try:
        # Tomar el primer resultado como base
        resultado_final = resultados[0].copy()
        
        # Combinar capítulos y artículos
        if 'estructura' in resultado_final:
            for resultado in resultados[1:]:
                if 'estructura' in resultado:
                    resultado_final['estructura']['capitulos'].extend(
                        resultado['estructura']['capitulos']
                    )
        
        # Combinar tags
        if 'tags' in resultado_final:
            tags_unicos = set(resultado_final['tags'])
            for resultado in resultados[1:]:
                if 'tags' in resultado:
                    tags_unicos.update(resultado['tags'])
            resultado_final['tags'] = list(tags_unicos)
        
        return resultado_final
        
    except Exception as e:
        logger.error(f"Error combinando resultados: {str(e)}")
        return resultados[0]  # Retornar el primer resultado si hay error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directorio con PDFs a procesar')
    parser.add_argument('--output-dir', required=True, help='Directorio para guardar resultados')
    parser.add_argument('--endpoint', required=True, help='Nombre del endpoint de SageMaker')
    parser.add_argument('--model', required=True, help='Nombre del modelo a usar (ej: llama-2-13b-chat)')
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lista de resultados para el CSV
    results = []
    pdf_count = 0
    success_count = 0
    error_count = 0
    
    # Procesar cada PDF
    for pdf_file in glob.glob(os.path.join(args.input_dir, '*.pdf')):
        pdf_count += 1
        filename = os.path.basename(pdf_file)
        
        if pdf_count <= 15:
            logging.info(f'Procesando {filename}...')
        else:
            if pdf_count % 10 == 0:  # Mostrar resumen cada 10 PDFs
                logging.info(f'Procesados {pdf_count} PDFs: {success_count} exitosos, {error_count} errores')
        
        try:
            # Extraer texto del PDF
            text = extract_text_from_pdf(pdf_file)
            
            # Limpiar texto
            text = clean_text(text)
            
            # Verificar longitud del texto
            if len(text) > 3000:  # Si el texto es muy largo
                if pdf_count <= 15:
                    logger.info(f"Texto demasiado largo ({len(text)} caracteres). Procesando por secciones...")
                result = procesar_texto_largo(text, args.endpoint, args.model, pdf_count)
            else:
                result = invoke_llama_model(text, args.endpoint, args.model, pdf_count)
            
            # Verificar si hubo error en la invocación
            if result is None:
                error_count += 1
                if pdf_count <= 15:
                    logging.error(f'Error en la respuesta del modelo para {filename}')
                continue
                
            # Validar estructura
            if not validate_structure(result):
                error_count += 1
                if pdf_count <= 15:
                    logging.error(f'Estructura inválida en respuesta para {filename}')
                continue
                
            success_count += 1
            
            # Guardar JSON individual
            output_json = os.path.join(args.output_dir, f'{os.path.splitext(filename)[0]}.json')
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Agregar a resultados para CSV
            results.append({
                'archivo': filename,
                'titulo': result['titulo'],
                'fecha': result['fecha'],
                'resumen': result['resumen']
            })
            
        except Exception as e:
            error_count += 1
            if pdf_count <= 15:
                logging.error(f'Error procesando {filename}: {str(e)}')
            continue
    
    # Mostrar resumen final
    logging.info(f'Procesamiento completado: {pdf_count} PDFs totales, {success_count} exitosos, {error_count} errores')
    
    # Guardar CSV con todos los resultados
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, 'resultados.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
    else:
        logging.warning('No se procesó ningún documento correctamente')

if __name__ == '__main__':
    main() 