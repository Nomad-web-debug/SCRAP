import json
import os
import boto3
import logging
from datetime import datetime
from typing import Dict, Optional
import fitz  # PyMuPDF
from io import BytesIO

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Inicializar clientes AWS
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configurar tabla DynamoDB
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])

def extract_text_from_pdf(pdf_content: bytes) -> Optional[str]:
    """
    Extrae texto de un PDF usando PyMuPDF
    """
    try:
        # Crear documento PDF en memoria
        pdf_stream = BytesIO(pdf_content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        # Extraer texto de cada página
        text_parts = []
        for page in doc:
            # Extraer texto con formato mejorado
            text = page.get_text("text")
            if text:
                text_parts.append(text)
        
        # Cerrar documento
        doc.close()
        
        return '\n'.join(text_parts)
        
    except Exception as e:
        logger.error(f"Error extrayendo texto del PDF: {str(e)}")
        return None

def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto extraído
    """
    # Eliminar caracteres especiales y espacios múltiples
    text = ' '.join(text.split())
    
    # Normalizar saltos de línea
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Normalizar espacios alrededor de puntuación
    text = text.replace(' .', '.').replace(' ,', ',')
    
    return text

def extract_basic_metadata(text: str, filename: str) -> Dict:
    """
    Extrae metadata básica del texto y nombre de archivo
    """
    metadata = {
        'archivo_original': filename,
        'fecha_extraccion': datetime.now().isoformat(),
        'tamanio_texto': len(text),
        'numero_palabras': len(text.split())
    }
    
    # Intentar extraer tipo de documento del nombre
    if 'ley' in filename.lower():
        metadata['tipo_documento_preliminar'] = 'LEY'
    elif 'decreto' in filename.lower():
        metadata['tipo_documento_preliminar'] = 'DECRETO'
    elif 'resolucion' in filename.lower():
        metadata['tipo_documento_preliminar'] = 'RESOLUCIÓN'
    else:
        metadata['tipo_documento_preliminar'] = 'OTRO'
    
    return metadata

def save_preprocessed(text: str, metadata: Dict, key: str) -> bool:
    """
    Guarda el texto preprocesado y su metadata
    """
    try:
        # Preparar datos
        data = {
            'texto': text,
            'metadata': metadata,
            'fecha_preprocesamiento': datetime.now().isoformat()
        }
        
        # Guardar en S3
        processed_bucket = os.environ['PROCESSED_BUCKET']
        s3.put_object(
            Bucket=processed_bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False),
            ContentType='application/json'
        )
        
        # Guardar referencia en DynamoDB
        table.put_item(Item={
            'id': f"PRE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'archivo_original': metadata['archivo_original'],
            'estado': 'PREPROCESADO',
            'fecha_preprocesamiento': data['fecha_preprocesamiento'],
            's3_location': f"s3://{processed_bucket}/{key}"
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Error guardando preprocesamiento: {str(e)}")
        return False

def lambda_handler(event, context):
    """
    Manejador principal de la función Lambda
    """
    try:
        # Obtener información del PDF de S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Verificar que es un PDF
        if not key.lower().endswith('.pdf'):
            logger.warning(f"Archivo no es PDF: {key}")
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'El archivo no es un PDF'})
            }
        
        # Obtener el PDF
        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = response['Body'].read()
        
        # Extraer texto
        text = extract_text_from_pdf(pdf_content)
        if not text:
            return {
                'statusCode': 500,
                'body': json.dumps({'message': 'Error extrayendo texto del PDF'})
            }
        
        # Limpiar texto
        text = clean_text(text)
        
        # Extraer metadata básica
        filename = os.path.basename(key)
        metadata = extract_basic_metadata(text, filename)
        
        # Generar key para archivo procesado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_key = f"preprocessed/{os.path.splitext(filename)[0]}_{timestamp}.json"
        
        # Guardar resultados
        if save_preprocessed(text, metadata, processed_key):
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'PDF preprocesado exitosamente',
                    'processed_key': processed_key
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({'message': 'Error guardando resultados'})
            }
            
    except Exception as e:
        logger.error(f"Error en lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error: {str(e)}'})
        } 