import boto3
import os
import logging
from datetime import datetime
import sys

# Configurar logging para cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def verify_s3_files():
    """Verifica los archivos en el bucket S3 antes del procesamiento"""
    try:
        # Verificar credenciales de AWS
        required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION', 'BUCKET_NAME']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Faltan variables de entorno requeridas: {', '.join(missing_vars)}")
            return False
        
        # Inicializar cliente S3 con región específica
        s3 = boto3.client(
            's3',
            region_name=os.getenv('AWS_DEFAULT_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        bucket = os.getenv('BUCKET_NAME')
        
        # Verificar que el bucket existe y es accesible
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception as e:
            logger.error(f"No se puede acceder al bucket {bucket}: {str(e)}")
            return False
        
        # Verificar archivos raw
        logger.info(f"Verificando archivos en el bucket {bucket}...")
        try:
            raw_objects = s3.list_objects_v2(Bucket=bucket, Prefix='raw/')
        except Exception as e:
            logger.error(f"Error listando objetos en el bucket: {str(e)}")
            return False
        
        if 'Contents' not in raw_objects:
            logger.error("No se encontraron archivos en el directorio raw/")
            return False
        
        # Filtrar solo archivos PDF
        pdf_files = [obj for obj in raw_objects['Contents'] if obj['Key'].lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error("No se encontraron archivos PDF para procesar")
            return False
        
        # Mostrar información de los archivos
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar:")
        for file in pdf_files:
            size_mb = file['Size'] / (1024 * 1024)
            last_modified = file['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"- {file['Key']} ({size_mb:.2f} MB, modificado: {last_modified})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verificando archivos en S3: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if verify_s3_files():
            logger.info("Verificación completada exitosamente")
            sys.exit(0)
        else:
            logger.error("La verificación falló")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        sys.exit(1) 