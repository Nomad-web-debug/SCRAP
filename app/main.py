import os
import json
import boto3
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import torch
from transformers import pipeline

# Cargar configuración
with open('config.json') as f:
    config = json.load(f)

# Inicializar clientes AWS
s3 = boto3.client('s3')
db_engine = create_engine(f"postgresql://dbadmin:{os.getenv('DB_PASSWORD')}@{config['rds_endpoint']}/clasificador")

# Inicializar modelo de clasificación
classifier = pipeline("text-classification", model="bert-base-uncased")

def process_document(file_path, file_name):
    """Procesa un documento PDF y extrae su información"""
    try:
        # Leer documento
        with open(file_path, 'rb') as file:
            content = file.read()
        
        # Subir a S3
        s3_key = f"raw/{datetime.now().strftime('%Y/%m/%d')}/{file_name}"
        s3.upload_fileobj(file, config['s3_bucket'], s3_key)
        
        # Extraer texto (ejemplo simple)
        text = extract_text(content)
        
        # Clasificar texto
        classification = classifier(text[:512])[0]
        
        # Guardar metadatos en RDS
        metadata = {
            'filename': file_name,
            's3_path': s3_key,
            'classification': classification['label'],
            'confidence': classification['score'],
            'processed_at': datetime.now()
        }
        
        df = pd.DataFrame([metadata])
        df.to_sql('documents', db_engine, if_exists='append', index=False)
        
        return {
            'status': 'success',
            'metadata': metadata
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def extract_text(content):
    """Extrae texto de un documento (implementar según necesidad)"""
    # TODO: Implementar extracción de texto según el tipo de documento
    return "Texto extraído del documento"

if __name__ == "__main__":
    # Código para procesar documentos en batch
    input_dir = "documents/input"
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(input_dir, filename)
            result = process_document(file_path, filename)
            print(f"Processed {filename}: {result['status']}") 