import json
import boto3
import os
from scraper import AIClassifier, DocumentProcessor
from io import BytesIO

s3 = boto3.client('s3')
sqs = boto3.client('sqs')

def lambda_handler(event, context):
    # Obtener el bucket y la key del archivo PDF del evento
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    try:
        # Descargar el PDF de S3
        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = response['Body'].read()
        
        # Procesar el PDF
        processor = DocumentProcessor()
        categories = ["categoria1", "categoria2", "categoria3"]  # Definir tus categorías aquí
        
        # Crear un archivo temporal para el PDF
        temp_pdf_path = '/tmp/temp.pdf'
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_content)
        
        # Clasificar el documento
        result = processor.process_pdf(temp_pdf_path, categories)
        
        # Guardar resultados en S3
        output_key = f"results/{key.split('/')[-1]}.json"
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(result)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps('Documento procesado exitosamente')
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        } 