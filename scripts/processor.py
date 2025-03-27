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

if __name__ == '__main__':
    processor = DocumentProcessor()
    processor.process_documents() 