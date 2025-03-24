import os
import boto3
from botocore.exceptions import ClientError
import json

class S3Manager:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = os.environ['S3_BUCKET']
    
    def save_document(self, doc_id, content, content_type='text/plain'):
        """Guarda un documento en S3"""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f'documents/{doc_id}',
                Body=content,
                ContentType=content_type
            )
            return True
        except ClientError as e:
            print(f"Error guardando documento: {e}")
            return False
    
    def save_structured_data(self, doc_id, data):
        """Guarda datos estructurados en formato JSON"""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f'structured/{doc_id}.json',
                Body=json.dumps(data),
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            print(f"Error guardando datos estructurados: {e}")
            return False
    
    def save_training_data(self, data, filename):
        """Guarda datos de entrenamiento"""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f'training/{filename}.json',
                Body=json.dumps(data),
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            print(f"Error guardando datos de entrenamiento: {e}")
            return False
    
    def get_document(self, doc_id):
        """Obtiene un documento de S3"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=f'documents/{doc_id}'
            )
            return response['Body'].read()
        except ClientError as e:
            print(f"Error obteniendo documento: {e}")
            return None
    
    def get_structured_data(self, doc_id):
        """Obtiene datos estructurados"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=f'structured/{doc_id}.json'
            )
            return json.loads(response['Body'].read())
        except ClientError as e:
            print(f"Error obteniendo datos estructurados: {e}")
            return None
    
    def get_training_data(self, filename):
        """Obtiene datos de entrenamiento"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=f'training/{filename}.json'
            )
            return json.loads(response['Body'].read())
        except ClientError as e:
            print(f"Error obteniendo datos de entrenamiento: {e}")
            return None 