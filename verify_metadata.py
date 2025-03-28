import boto3
import os
import json
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_metadata():
    try:
        s3 = boto3.client('s3')
        bucket = os.getenv('BUCKET_NAME', 'clasificador-docs-13dgv6lo')
        
        logger.info(f"Verificando metadata en bucket: {bucket}")
        
        # Verificar archivos nuevos (últimos 5 minutos)
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        
        # Verificar metadata
        meta_response = s3.list_objects_v2(Bucket=bucket, Prefix='metadata/')
        if 'Contents' not in meta_response:
            logger.error('No se encontraron archivos de metadata')
            return False
            
        recent_meta = [obj for obj in meta_response['Contents'] 
                      if obj['LastModified'].replace(tzinfo=None) > cutoff]
        
        if not recent_meta:
            logger.error('No se encontraron metadatos recientes')
            return False
        
        # Leer último archivo de metadata
        latest_meta = max(recent_meta, key=lambda x: x['LastModified'])
        logger.info(f"Último archivo de metadata: {latest_meta['Key']}")
        
        content = s3.get_object(Bucket=bucket, Key=latest_meta['Key'])
        data = json.loads(content['Body'].read())
        
        total_docs = data.get('total', 0)
        logger.info(f'Documentos encontrados: {total_docs}')
        
        if total_docs == 0:
            logger.error('No se encontraron documentos')
            return False
            
        logger.info('Verificación completada exitosamente')
        return True
        
    except Exception as e:
        logger.error(f"Error durante la verificación: {str(e)}")
        return False

if __name__ == '__main__':
    success = verify_metadata()
    exit(0 if success else 1) 