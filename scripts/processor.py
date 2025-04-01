import boto3
import json
import os
from datetime import datetime
import logging
import anthropic
from PyPDF2 import PdfReader
from io import BytesIO
import uuid
import time
import sys
import argparse
import pandas as pd
from text_structure import TextStructureProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.claude_client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.bucket_name = os.getenv('BUCKET_NAME')

    def extract_text_from_pdf(self, pdf_key):
        """Extrae texto de un PDF almacenado en S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=pdf_key)
            pdf_content = BytesIO(response['Body'].read())
            pdf_reader = PdfReader(pdf_content)
            
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF {pdf_key}: {str(e)}")
            return None

    def analyze_with_claude(self, text, metadata):
        """Analiza el texto con Claude para extraer información estructurada"""
        try:
            text_sample = text[:8000] if len(text) > 8000 else text
            
            prompt = f'''Analiza el siguiente texto legal y extrae la información solicitada.
            
            Información existente del documento:
            - Título: {metadata.get('titulo', 'No disponible')}
            - Número de Norma: {metadata.get('nro_norma', 'No disponible')}
            - Materia actual: {metadata.get('materia', 'No disponible')}
            
            Basándote en el contenido del texto y la información existente, genera un JSON con los siguientes campos:
            {{
                "titulo": "mantener el título original",
                "nro_norma": "mantener el número original",
                "categoria_principal": {{
                    "valor": "una de: penal, civil, laboral, tributario, ambiental",
                    "confianza": "porcentaje de 0 a 100"
                }},
                "subcategorias": [
                    {{"valor": "subcategoría", "confianza": "porcentaje"}}
                ],
                "materia": "mantener la materia original",
                "palabras_clave": ["máximo", "5", "palabras"],
                "resumen_ejecutivo": "resumen detallado que capture los puntos principales",
                "justificacion": "explicación de por qué se eligió esta categorización"
            }}
            
            Es crucial mantener los valores originales de título, número de norma y materia.
            Solo debes categorizar y añadir información adicional.
            
            Texto a analizar:
            {text_sample}'''
            
            response = self.claude_client.messages.create(
                model='claude-3-sonnet-20240229',
                max_tokens=1000,
                temperature=0,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            analysis = json.loads(response.content)
            
            # Registrar métricas de confianza
            self._log_confidence_metrics(analysis)
            
            return analysis
                
        except Exception as e:
            logger.error(f"Error analizando con Claude: {str(e)}")
            return None

    def _log_confidence_metrics(self, analysis):
        """Registra métricas de confianza de la categorización"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "categoria_confianza": analysis["categoria_principal"]["confianza"],
                "subcategorias_confianza": [sub["confianza"] for sub in analysis["subcategorias"]],
                "tiene_justificacion": bool(analysis.get("justificacion")),
                "titulo_original": analysis.get("titulo"),
                "nro_norma_original": analysis.get("nro_norma"),
                "materia_original": analysis.get("materia")
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"metrics/confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                Body=json.dumps(metrics),
                ContentType='application/json'
            )
            
            if float(analysis["categoria_principal"]["confianza"]) < 70:
                logger.warning(f"Baja confianza en categorización: {analysis['justificacion']}")
                
        except Exception as e:
            logger.error(f"Error registrando métricas: {str(e)}")

    def process_documents(self):
        """Procesa los documentos pendientes"""
        try:
            # Obtener metadatos
            meta_response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='metadata/'
            )
            
            if 'Contents' not in meta_response:
                logger.info("No hay metadatos para procesar")
                return
            
            # Obtener el archivo de metadatos más reciente
            latest_meta = max(meta_response['Contents'], key=lambda x: x['LastModified'])
            meta_content = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=latest_meta['Key']
            )
            metadata = json.loads(meta_content['Body'].read())
            
            # Procesar cada documento
            for doc in metadata.get('documentos', []):
                try:
                    # Construir nombre del archivo PDF
                    pdf_filename = f"{doc['nro_norma']}_{doc['titulo']}"[:100]
                    pdf_filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in pdf_filename)
                    pdf_key = f"raw/{pdf_filename}.pdf"
                    
                    # Verificar si ya fue procesado
                    processed_key = f"processed/{pdf_filename}.json"
                    try:
                        self.s3_client.head_object(
                            Bucket=self.bucket_name,
                            Key=processed_key
                        )
                        logger.info(f"Documento ya procesado: {pdf_filename}")
                        continue
                    except:
                        pass
                    
                    # Extraer texto
                    text = self.extract_text_from_pdf(pdf_key)
                    if not text:
                        continue
                    
                    # Analizar con Claude
                    analysis = self.analyze_with_claude(text, doc)
                    if analysis:
                        # Guardar resultados
                        analysis['fecha_procesamiento'] = datetime.now().isoformat()
                        analysis['pdf_original'] = pdf_key
                        
                        self.s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=processed_key,
                            Body=json.dumps(analysis, ensure_ascii=False),
                            ContentType='application/json'
                        )
                        logger.info(f"Documento procesado: {pdf_filename}")
                    
                except Exception as e:
                    logger.error(f"Error procesando documento {doc.get('titulo', 'desconocido')}: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error en el procesamiento: {str(e)}")

def save_to_csv(results, output_dir):
    """Guarda los resultados en formato CSV para entrenamiento"""
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
                'texto_completo': result.get('texto_norma', ''),
                'palabras_clave': '|'.join(result.get('palabras_clave', [])),
                'tipo_norma': result.get('tipo_norma', ''),
                'numero_norma': result.get('numero_norma', ''),
                'entidad_emisora': result.get('entidad_emisora', ''),
                'ambito_aplicacion': result.get('ambito_aplicacion', ''),
                'estado_vigencia': result.get('estado_vigencia', ''),
                'fecha_procesamiento': result.get('fecha_procesamiento', ''),
                'archivo_original': result.get('archivo_original', '')
            }
            
            # Añadir referencias normativas
            referencias = result.get('referencias_normativas', [])
            row['referencias_normativas'] = '|'.join(referencias)
            
            # Añadir modificaciones
            modificaciones = result.get('modificaciones', [])
            row['modificaciones'] = '|'.join(modificaciones)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Guardar CSV localmente
        csv_filename = f"datos_entrenamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Datos guardados en CSV: {csv_path}")
        
        # Guardar CSV en S3
        s3 = boto3.client('s3')
        bucket = os.getenv('BUCKET_NAME')
        s3_key = f"processed_local/{csv_filename}"
        
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=df.to_csv(index=False).encode('utf-8'),
            ContentType='text/csv'
        )
        print(f"CSV guardado en S3: {s3_key}")
        
    except Exception as e:
        print(f"Error guardando CSV: {str(e)}")

def process_local_pdfs(input_dir, output_dir):
    """Procesa PDFs desde una carpeta local"""
    processor = TextStructureProcessor()
    
    # Verificar que la carpeta existe
    if not os.path.exists(input_dir):
        print(f"Error: La carpeta {input_dir} no existe")
        sys.exit(1)
        
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
        
    # Obtener lista de PDFs
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No se encontraron archivos PDF en la carpeta {input_dir}")
        sys.exit(1)
        
    print(f"Encontrados {len(pdf_files)} archivos PDF para procesar")
    
    # Lista para almacenar todos los resultados
    all_results = []
    
    # Procesar cada PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        print(f"\nProcesando: {pdf_file}")
        
        try:
            # Extraer texto del PDF
            pdf_reader = PdfReader(pdf_path)
            texto_completo = []
            for page in pdf_reader.pages:
                texto_completo.append(page.extract_text())
            texto_completo = '\n'.join(texto_completo)
            
            # Procesar el documento
            result = processor.process_document(pdf_path)
            
            # Añadir información adicional
            result['fecha_procesamiento'] = datetime.now().isoformat()
            result['archivo_original'] = pdf_file
            result['estado_vigencia'] = 'VIGENTE'  # Por defecto
            result['ambito_aplicacion'] = 'NACIONAL'  # Por defecto
            result['texto_norma'] = texto_completo  # Añadido texto completo
            
            # Crear nombre único para el archivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{timestamp}_{os.path.splitext(pdf_file)[0]}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            # Guardar resultado localmente
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Documento procesado y guardado localmente: {output_path}")
            
            # Guardar resultado en S3
            s3 = boto3.client('s3')
            bucket = os.getenv('BUCKET_NAME')
            s3_key = f"processed_local/{output_filename}"
            
            s3.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json.dumps(result, ensure_ascii=False),
                ContentType='application/json'
            )
            
            print(f"Documento guardado en S3: {s3_key}")
            
            # Añadir resultado a la lista
            all_results.append(result)
            
        except Exception as e:
            print(f"Error procesando {pdf_file}: {str(e)}")
            continue
    
    # Guardar todos los resultados en CSV
    if all_results:
        save_to_csv(all_results, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Procesador de documentos PDF')
    parser.add_argument('--input-dir', help='Carpeta con PDFs a procesar')
    parser.add_argument('--output-dir', help='Carpeta para guardar resultados estructurados')
    args = parser.parse_args()
    
    if args.input_dir:
        output_dir = args.output_dir or os.getenv('OUTPUT_DIR', 'estructurados_local')
        process_local_pdfs(args.input_dir, output_dir)
    else:
        # Código existente para procesar desde S3
        processor = DocumentProcessor()
        processor.process_documents()

if __name__ == "__main__":
    main() 