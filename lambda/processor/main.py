import json
import os
import boto3
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sagemaker.predictor

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Inicializar clientes AWS
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Configurar tabla DynamoDB
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])

def generate_prompt(text: str) -> str:
    """
    Genera el prompt para el modelo Llama
    """
    return f"""Analiza el siguiente texto legal y extrae la información estructurada según este formato:

    FORMATO REQUERIDO:
    {{
        "id": "identificador-único",
        "documento": "nombre del documento",
        "tipo_documento": "tipo de norma",
        "rama_derecho": "rama jurídica",
        "articulo_numero": "número del artículo",
        "articulo_titulo": "título del artículo",
        "contenido": "texto completo del artículo",
        "titulo": "título mayor (opcional)",
        "capitulo": "capítulo específico (opcional)",
        "tags": ["palabras", "clave"],
        "fuente": "referencia",
        "modificado": false,
        "fecha_ultima_actualizacion": null,
        "comentario_IA": "explicación para usuarios no expertos"
    }}

    TEXTO A ANALIZAR:
    {text}

    INSTRUCCIONES ESPECÍFICAS:
    1. Mantén el formato JSON exacto
    2. Asegúrate de que todos los campos obligatorios estén presentes
    3. Genera un ID único basado en el tipo y número de documento
    4. Identifica correctamente la rama del derecho
    5. Extrae con precisión los números y títulos de artículos
    6. Genera tags relevantes
    7. Añade un comentario explicativo claro y conciso
    """

def invoke_llama(text: str) -> Dict:
    """
    Invoca el endpoint de Llama para procesar el texto
    """
    try:
        endpoint_name = os.environ['SAGEMAKER_ENDPOINT']
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                'prompt': generate_prompt(text),
                'max_tokens': 2000,
                'temperature': 0.1,
                'top_p': 0.9
            })
        )
        
        result = json.loads(response['Body'].read().decode())
        return result
        
    except Exception as e:
        logger.error(f"Error invocando Llama: {str(e)}")
        raise

def validate_structure(data: Dict) -> bool:
    """
    Valida que la estructura tenga todos los campos requeridos
    """
    required_fields = [
        'id', 'documento', 'tipo_documento', 'rama_derecho',
        'articulo_numero', 'articulo_titulo', 'contenido'
    ]
    
    return all(field in data for field in required_fields)

def save_to_dynamodb(data: Dict) -> bool:
    """
    Guarda los datos estructurados en DynamoDB
    """
    try:
        # Añadir timestamp
        data['fecha_procesamiento'] = datetime.now().isoformat()
        
        # Guardar en DynamoDB
        table.put_item(Item=data)
        return True
        
    except Exception as e:
        logger.error(f"Error guardando en DynamoDB: {str(e)}")
        return False

def save_to_s3(data: Dict, key: str) -> bool:
    """
    Guarda los datos estructurados en S3
    """
    try:
        bucket = os.environ['PROCESSED_BUCKET']
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False),
            ContentType='application/json'
        )
        return True
        
    except Exception as e:
        logger.error(f"Error guardando en S3: {str(e)}")
        return False

def process_document(text: str, metadata: Dict) -> Optional[Dict]:
    """
    Procesa un documento usando Llama y guarda los resultados
    """
    try:
        # Invocar Llama para estructurar el texto
        result = invoke_llama(text)
        
        # Validar estructura
        if not validate_structure(result):
            logger.error("Estructura inválida en la respuesta de Llama")
            return None
            
        # Añadir metadata
        result.update(metadata)
        
        # Guardar en DynamoDB
        if not save_to_dynamodb(result):
            logger.error("Error guardando en DynamoDB")
            return None
            
        # Guardar en S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"processed/{result['id']}_{timestamp}.json"
        
        if not save_to_s3(result, s3_key):
            logger.error("Error guardando en S3")
            return None
            
        logger.info(f"Documento procesado exitosamente: {result['id']}")
        return result
        
    except Exception as e:
        logger.error(f"Error procesando documento: {str(e)}")
        return None

def lambda_handler(event, context):
    """
    Manejador principal de la función Lambda
    """
    try:
        # Obtener información del documento de S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Obtener el documento
        response = s3.get_object(Bucket=bucket, Key=key)
        text = response['Body'].read().decode('utf-8')
        
        # Obtener metadata del nombre del archivo
        filename = os.path.basename(key)
        metadata = {
            'archivo_original': filename,
            'bucket_origen': bucket
        }
        
        # Procesar documento
        result = process_document(text, metadata)
        
        if result:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Documento procesado exitosamente',
                    'id': result['id']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Error procesando documento'
                })
            }
            
    except Exception as e:
        logger.error(f"Error en lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error: {str(e)}'
            })
        } 