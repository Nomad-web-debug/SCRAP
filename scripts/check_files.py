import boto3
import os

s3 = boto3.client('s3')
bucket = os.getenv('BUCKET_NAME')

# Verificar archivos raw
raw = s3.list_objects_v2(Bucket=bucket, Prefix='raw/')
if not raw.get('Contents'):
    print('No hay archivos para procesar')
    exit(1)
    
print(f'Encontrados {len(raw["Contents"])} archivos para procesar') 