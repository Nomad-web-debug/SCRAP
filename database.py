import os
import boto3
from datetime import datetime

class DynamoDBManager:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.metadata_table = self.dynamodb.Table(os.environ['DYNAMODB_METADATA_TABLE'])
        self.training_table = self.dynamodb.Table(os.environ['DYNAMODB_TRAINING_TABLE'])
        self.chat_table = self.dynamodb.Table(os.environ['DYNAMODB_CHAT_TABLE'])
    
    def save_metadata(self, metadata):
        """Guarda metadata de un documento"""
        self.metadata_table.put_item(Item=metadata)
    
    def update_metadata(self, doc_id, updates):
        """Actualiza metadata de un documento"""
        update_expression = "SET "
        expression_values = {}
        
        for key, value in updates.items():
            update_expression += f"#{key} = :{key}, "
            expression_values[f":{key}"] = value
        
        update_expression = update_expression.rstrip(", ")
        
        self.metadata_table.update_item(
            Key={'id': doc_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames={f"#{k}": k for k in updates.keys()}
        )
    
    def get_processed_documents(self):
        """Obtiene todos los documentos procesados"""
        response = self.metadata_table.scan(
            FilterExpression='#estado = :estado',
            ExpressionAttributeNames={'#estado': 'estado'},
            ExpressionAttributeValues={':estado': 'procesado'}
        )
        return response['Items']
    
    def save_chat_message(self, session_id, user_message, ai_response):
        """Guarda un mensaje del chat"""
        timestamp = int(datetime.now().timestamp())
        
        self.chat_table.put_item(Item={
            'session_id': session_id,
            'timestamp': timestamp,
            'user_message': user_message,
            'ai_response': ai_response
        })
    
    def get_chat_history(self, session_id, limit=10):
        """Obtiene el historial de chat de una sesión"""
        response = self.chat_table.query(
            KeyConditionExpression='session_id = :sid',
            ExpressionAttributeValues={':sid': session_id},
            Limit=limit,
            ScanIndexForward=False  # Orden descendente por timestamp
        )
        
        # Convertir a formato para Claude
        messages = []
        for item in reversed(response['Items']):  # Revertir para orden cronológico
            messages.extend([
                {"role": "user", "content": item['user_message']},
                {"role": "assistant", "content": item['ai_response']}
            ])
        
        return messages 