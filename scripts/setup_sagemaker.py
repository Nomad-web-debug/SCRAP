import boto3
import json
import time
import logging
import os
from botocore.exceptions import ClientError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sagemaker_endpoint():
    """
    Crea un endpoint de SageMaker con Llama-2-70B si no existe
    """
    try:
        # Configurar región explícitamente
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Inicializar clientes con región específica
        session = boto3.Session(region_name=region)
        sagemaker = session.client('sagemaker')
        runtime = session.client('sagemaker-runtime')
        
        logger.info(f"Configurando servicios en la región: {region}")
        
        endpoint_name = 'llama-2-70b-endpoint'
        
        # Verificar si el endpoint ya existe
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"El endpoint {endpoint_name} ya existe")
            return endpoint_name
        except ClientError as e:
            if e.response['Error']['Code'] != 'ValidationException':
                raise
        
        # Crear el modelo
        model_name = 'llama-2-70b'
        try:
            sagemaker.describe_model(ModelName=model_name)
            logger.info(f"El modelo {model_name} ya existe")
        except ClientError:
            logger.info(f"Creando modelo {model_name}...")
            sagemaker.create_model(
                ModelName=model_name,
                ExecutionRoleArn='arn:aws:iam::aws:role/service-role/AmazonSageMaker-ExecutionRole',
                PrimaryContainer={
                    'Image': f'763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.0.1-transformers4.28.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
                    'ModelDataUrl': 's3://huggingface-sagemaker-models/llama-2-70b/model.tar.gz'
                }
            )
        
        # Crear configuración del endpoint
        endpoint_config_name = f'{endpoint_name}-config'
        try:
            sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            logger.info(f"La configuración {endpoint_config_name} ya existe")
        except ClientError:
            logger.info(f"Creando configuración del endpoint {endpoint_config_name}...")
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'InstanceType': 'ml.g5.12xlarge',
                    'InitialInstanceCount': 1,
                    'ModelName': model_name,
                    'VariantName': 'AllTraffic',
                    'ServerlessConfig': {
                        'MaxConcurrency': 1,
                        'MemorySizeInMB': 6144
                    }
                }]
            )
        
        # Crear el endpoint
        logger.info(f"Creando endpoint {endpoint_name}...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        # Esperar a que el endpoint esté listo
        while True:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status == 'InService':
                logger.info(f"Endpoint {endpoint_name} creado exitosamente")
                break
            elif status == 'Failed':
                raise Exception(f"Error creando endpoint: {response.get('FailureReason', 'Unknown error')}")
            time.sleep(30)
        
        return endpoint_name
        
    except Exception as e:
        logger.error(f"Error creando endpoint: {str(e)}")
        raise

def test_endpoint(endpoint_name):
    """
    Prueba el endpoint con un prompt simple
    """
    try:
        # Usar la misma región que el endpoint
        region = os.environ.get('AWS_REGION', 'us-east-1')
        runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                "prompt": "Hola, ¿cómo estás?",
                "max_tokens": 100
            })
        )
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Prueba exitosa: {result}")
        return True
    except Exception as e:
        logger.error(f"Error probando endpoint: {str(e)}")
        return False

if __name__ == '__main__':
    # Verificar región
    if 'AWS_REGION' not in os.environ:
        logger.error("AWS_REGION no está configurada")
        exit(1)
        
    try:
        endpoint_name = create_sagemaker_endpoint()
        if test_endpoint(endpoint_name):
            print(f"Endpoint {endpoint_name} configurado y funcionando correctamente")
        else:
            print("Error al probar el endpoint")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 