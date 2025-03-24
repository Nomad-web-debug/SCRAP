from anthropic import Anthropic
import json
from datetime import datetime

class DocumentProcessor:
    def __init__(self, anthropic_client):
        self.anthropic = anthropic_client
        self.token_usage = {
            'total_tokens': 0,
            'estimated_cost': 0,
            'last_reset': datetime.now().isoformat()
        }
        self.token_limit = 1000000  # Límite mensual de tokens
        self.cost_per_token = 0.00001  # Costo estimado por token
        
        # Cargar estado anterior si existe
        try:
            with open('token_usage.json', 'r') as f:
                self.token_usage = json.load(f)
        except FileNotFoundError:
            self.save_token_usage()
    
    def save_token_usage(self):
        """Guarda el uso de tokens actual"""
        with open('token_usage.json', 'w') as f:
            json.dump(self.token_usage, f)
    
    def check_token_limit(self, estimated_tokens):
        """Verifica si procesar más tokens excedería el límite"""
        if self.token_usage['total_tokens'] + estimated_tokens > self.token_limit:
            raise Exception("Se alcanzó el límite de tokens. Por favor, revise el uso y ajuste los límites si es necesario.")
    
    def update_token_usage(self, tokens_used):
        """Actualiza el contador de tokens y costos"""
        self.token_usage['total_tokens'] += tokens_used
        self.token_usage['estimated_cost'] += tokens_used * self.cost_per_token
        self.save_token_usage()
    
    def process_with_claude(self, content):
        """Procesa el contenido usando Claude para estructurarlo"""
        # Estimar tokens (aproximadamente 4 caracteres por token)
        estimated_tokens = len(content['contenido']) // 4
        self.check_token_limit(estimated_tokens)
        
        prompt = f"""
        Analiza el siguiente contenido y estructúralo en un formato JSON con las siguientes secciones:
        - Tema principal
        - Puntos clave
        - Análisis crítico
        - Recomendaciones
        - Fuentes o referencias mencionadas

        Contenido:
        {content['contenido']}
        """
        
        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Actualizar uso de tokens
        self.update_token_usage(estimated_tokens)
        
        # Extraer y validar JSON de la respuesta
        try:
            structured_data = response.content[0].text
            return {
                'original_content': content,
                'structured_data': structured_data,
                'metadata': {
                    'model': "claude-3-opus-20240229",
                    'timestamp': response.created_at,
                    'tokens_used': estimated_tokens
                }
            }
        except Exception as e:
            raise Exception(f"Error al procesar respuesta de Claude: {str(e)}")
    
    def prepare_training_data(self, content):
        """Prepara datos para entrenamiento del modelo de crítica/consejo"""
        prompt = f"""
        A partir del siguiente contenido estructurado, genera 5 ejemplos de entrenamiento
        para un modelo de IA que debe aprender a dar críticas constructivas y consejos.
        Cada ejemplo debe tener:
        - Contexto
        - Crítica constructiva
        - Consejo práctico
        - Justificación

        Contenido:
        {content['structured_data']}
        """
        
        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        try:
            training_examples = response.content[0].text
            return {
                'source_content': content['structured_data'],
                'training_examples': training_examples,
                'metadata': {
                    'model': "claude-3-opus-20240229",
                    'timestamp': response.created_at
                }
            }
        except Exception as e:
            raise Exception(f"Error al preparar datos de entrenamiento: {str(e)}") 