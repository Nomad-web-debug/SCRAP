import os
import json
import boto3
from anthropic import Anthropic
from datetime import datetime
from scraper import WebScraper
from processor import DocumentProcessor
from database import DynamoDBManager
from storage import S3Manager

# Inicializar clientes
anthropic = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
s3 = S3Manager()
db = DynamoDBManager()

def lambda_handler(event, context):
    """
    Función principal que maneja diferentes tipos de eventos:
    1. Scraping y procesamiento inicial
    2. Estructuración de datos con Claude
    3. Preparación de datos para entrenamiento
    4. Consultas al chatbot
    """
    try:
        # Determinar tipo de evento
        event_type = event.get('type', 'scrape')
        
        if event_type == 'scrape':
            return handle_scraping(event)
        elif event_type == 'process':
            return handle_processing(event)
        elif event_type == 'prepare_training':
            return handle_training_prep(event)
        elif event_type == 'chat':
            return handle_chat(event)
        else:
            raise ValueError(f"Tipo de evento no soportado: {event_type}")
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def handle_scraping(event):
    """Maneja el scraping de documentos y los guarda en S3"""
    scraper = WebScraper()
    urls = event.get('urls', [])
    results = []
    
    for url in urls:
        # Realizar scraping
        content = scraper.scrape(url)
        
        # Guardar en S3
        file_key = f"raw/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{url.split('/')[-1]}.json"
        s3.upload_json(content, file_key)
        
        # Guardar metadata en DynamoDB
        metadata = {
            'id': file_key,
            'fecha': datetime.now().isoformat(),
            'url': url,
            'tipo': 'scraping',
            'estado': 'pendiente_procesar'
        }
        db.save_metadata(metadata)
        
        results.append({
            'url': url,
            'status': 'success',
            'file_key': file_key
        })
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def handle_processing(event):
    """Procesa documentos usando Claude para estructuración"""
    file_keys = event.get('file_keys', [])
    processor = DocumentProcessor(anthropic)
    results = []
    
    for file_key in file_keys:
        # Obtener documento de S3
        content = s3.get_json(file_key)
        
        # Procesar con Claude
        structured_data = processor.process_with_claude(content)
        
        # Guardar resultados procesados
        processed_key = f"processed/{file_key.split('/')[-1]}"
        s3.upload_json(structured_data, processed_key)
        
        # Actualizar metadata
        db.update_metadata(file_key, {
            'estado': 'procesado',
            'processed_file': processed_key
        })
        
        results.append({
            'file_key': file_key,
            'status': 'processed',
            'processed_key': processed_key
        })
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def handle_training_prep(event):
    """Prepara datos para entrenamiento del modelo de crítica/consejo"""
    processor = DocumentProcessor(anthropic)
    
    # Obtener todos los documentos procesados
    processed_docs = db.get_processed_documents()
    
    training_data = []
    for doc in processed_docs:
        # Obtener documento procesado
        content = s3.get_json(doc['processed_file'])
        
        # Preparar datos para entrenamiento
        training_examples = processor.prepare_training_data(content)
        training_data.extend(training_examples)
    
    # Guardar datos de entrenamiento
    training_file = f"training/dataset_{datetime.now().strftime('%Y%m%d')}.json"
    s3.upload_json(training_data, training_file)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'training_file': training_file,
            'examples_count': len(training_data)
        })
    }

def handle_chat(event):
    """Maneja interacciones con el chatbot"""
    message = event.get('message', '')
    session_id = event.get('session_id', '')
    
    # Obtener historial de chat si existe
    chat_history = db.get_chat_history(session_id)
    
    # Procesar mensaje con Claude
    response = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            *chat_history,
            {"role": "user", "content": message}
        ]
    )
    
    # Guardar en historial
    db.save_chat_message(session_id, message, response.content[0].text)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'response': response.content[0].text
        })
    } 