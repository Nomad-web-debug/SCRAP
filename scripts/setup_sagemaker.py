import boto3
import json
import time
import logging
import os
from botocore.exceptions import ClientError
import argparse
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de modelos disponibles
MODELO_CONFIG = {
    '7b': {
        'nombre': 'llama-2-7b',
        'modelo_hf': 'meta-llama/Llama-2-7b-chat-hf',
        'instancia': 'ml.g5.xlarge',
        'descripcion': 'Modelo más ligero y económico'
    },
    '13b': {
        'nombre': 'llama-2-13b',
        'modelo_hf': 'meta-llama/Llama-2-13b-chat-hf',
        'instancia': 'ml.g5.2xlarge',
        'descripcion': 'Balance entre rendimiento y costo'
    },
    '70b': {
        'nombre': 'llama-2-70b',
        'modelo_hf': 'meta-llama/Llama-2-70b-chat-hf',
        'instancia': 'ml.g5.12xlarge',
        'descripcion': 'Modelo más potente y preciso'
    }
}

def get_or_create_role(iam_client):
    """
    Obtiene o crea el rol de ejecución para SageMaker
    """
    role_name = 'AmazonSageMaker-ExecutionRole'
    
    try:
        # Intentar obtener el rol
        response = iam_client.get_role(RoleName=role_name)
        logger.info(f"Rol {role_name} encontrado")
        
        # Asegurar que tenga los permisos necesarios
        attach_required_policies(iam_client, role_name)
        
        return response['Role']['Arn']
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            # Crear el rol si no existe
            logger.info(f"Creando rol {role_name}...")
            
            # Política de confianza para SageMaker
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "sagemaker.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            try:
                # Crear el rol
                response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy)
                )
                
                # Adjuntar políticas necesarias
                attach_required_policies(iam_client, role_name)
                
                logger.info(f"Rol {role_name} creado exitosamente")
                return response['Role']['Arn']
                
            except ClientError as create_error:
                if create_error.response['Error']['Code'] == 'EntityAlreadyExists':
                    # Si el rol ya existe mientras intentamos crearlo, intentar obtenerlo de nuevo
                    response = iam_client.get_role(RoleName=role_name)
                    logger.info(f"Usando rol {role_name} existente")
                    
                    # Asegurar que tenga los permisos necesarios
                    attach_required_policies(iam_client, role_name)
                    
                    return response['Role']['Arn']
                else:
                    raise create_error
        else:
            raise

def attach_required_policies(iam_client, role_name):
    """
    Adjunta todas las políticas necesarias al rol
    """
    required_policies = [
        'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly'
    ]
    
    # Política inline para acceder a SageMaker y ECR
    sagemaker_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:*",
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "iam:PassRole",
                    "iam:CreateRole",
                    "iam:DeleteRole"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/AmazonSageMaker*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::sagemaker-*",
                    "arn:aws:s3:::huggingface-*"
                ]
            }
        ]
    }
    
    # Adjuntar políticas AWS administradas
    for policy_arn in required_policies:
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            logger.info(f"Política {policy_arn} adjuntada al rol")
        except ClientError as e:
            if e.response['Error']['Code'] != 'EntityAlreadyExists':
                raise
    
    # Adjuntar política inline para SageMaker y ECR
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName='SageMakerAndECRAccess',
            PolicyDocument=json.dumps(sagemaker_policy)
        )
        logger.info("Política inline para SageMaker y ECR adjuntada al rol")
    except Exception as e:
        logger.error(f"Error adjuntando política inline: {str(e)}")
        raise

def eliminar_endpoint_existente(sagemaker, endpoint_name):
    """
    Elimina un endpoint existente si existe
    """
    try:
        logger.info(f"Verificando si existe el endpoint {endpoint_name}...")
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Eliminando endpoint existente {endpoint_name}...")
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        sagemaker.get_waiter('endpoint_deleted').wait(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} eliminado correctamente")
    except ClientError as e:
        if e.response['Error']['Code'] != 'ValidationException':
            raise

def cleanup_resources(sagemaker, model_name):
    """
    Limpia todos los recursos asociados en caso de error
    """
    try:
        # Eliminar endpoint si existe
        endpoint_name = f"{model_name}-endpoint"
        try:
            logger.info(f"Eliminando endpoint {endpoint_name}...")
            sagemaker.delete_endpoint(EndpointName=endpoint_name)
            waiter = sagemaker.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} eliminado")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"Error eliminando endpoint: {str(e)}")

        # Eliminar configuración del endpoint
        config_name = f"{model_name}-config"
        try:
            logger.info(f"Eliminando configuración {config_name}...")
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            logger.info(f"Configuración {config_name} eliminada")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"Error eliminando configuración: {str(e)}")

        # Eliminar modelo
        try:
            logger.info(f"Eliminando modelo {model_name}...")
            sagemaker.delete_model(ModelName=model_name)
            logger.info(f"Modelo {model_name} eliminado")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"Error eliminando modelo: {str(e)}")

        # Esperar un momento para asegurar que AWS ha liberado los recursos
        logger.info("Esperando a que AWS libere los recursos...")
        time.sleep(30)

    except Exception as e:
        logger.error(f"Error durante la limpieza: {str(e)}")

def save_endpoint_state(endpoint_name: str, modelo: str):
    """
    Guarda el estado del endpoint activo
    """
    state = {
        'endpoint_name': endpoint_name,
        'modelo': modelo,
        'ultima_actualizacion': datetime.now().isoformat()
    }
    
    with open('endpoint_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info(f"Estado del endpoint guardado en endpoint_state.json")

def get_active_endpoint():
    """
    Obtiene el endpoint activo desde el archivo de estado
    """
    try:
        with open('endpoint_state.json', 'r') as f:
            state = json.load(f)
            return state.get('endpoint_name'), state.get('modelo')
    except FileNotFoundError:
        return None, None

def list_active_endpoints(sagemaker):
    """
    Lista todos los endpoints activos
    """
    try:
        response = sagemaker.list_endpoints()
        return [endpoint['EndpointName'] for endpoint in response['Endpoints']]
    except Exception as e:
        logger.error(f"Error listando endpoints: {str(e)}")
        return []

def list_all_resources(sagemaker):
    """
    Lista todos los recursos de SageMaker
    """
    resources = {
        'endpoints': [],
        'endpoint_configs': [],
        'models': []
    }
    
    try:
        # Listar endpoints
        response = sagemaker.list_endpoints()
        resources['endpoints'] = [endpoint['EndpointName'] for endpoint in response['Endpoints']]
        
        # Listar configuraciones
        response = sagemaker.list_endpoint_configs()
        resources['endpoint_configs'] = [config['EndpointConfigName'] for config in response['EndpointConfigs']]
        
        # Listar modelos
        response = sagemaker.list_models()
        resources['models'] = [model['ModelName'] for model in response['Models']]
        
    except Exception as e:
        logger.error(f"Error listando recursos: {str(e)}")
    
    return resources

def create_sagemaker_endpoint(modelo_elegido='13b', force_recreate=False):
    """
    Crea un endpoint de SageMaker con el modelo Llama 2 especificado
    """
    try:
        # Verificar que el token está configurado
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_HUB_TOKEN no está configurado. Por favor, configura la variable de entorno.")

        # Obtener configuración del modelo
        if modelo_elegido not in MODELO_CONFIG:
            raise ValueError(f"Modelo no válido. Opciones disponibles: {', '.join(MODELO_CONFIG.keys())}")
        
        config = MODELO_CONFIG[modelo_elegido]
        logger.info(f"Configurando modelo {config['nombre']} ({config['descripcion']})")
        
        # Configurar región y servicios
        region = os.environ.get('AWS_REGION', 'us-east-1')
        session = boto3.Session(region_name=region)
        sagemaker = session.client('sagemaker')
        iam = session.client('iam')
        
        # Obtener o crear rol
        role_arn = get_or_create_role(iam)
        
        # Listar y eliminar todos los recursos existentes
        logger.info("Verificando recursos existentes...")
        resources = list_all_resources(sagemaker)
        
        # Eliminar todos los endpoints
        for endpoint in resources['endpoints']:
            logger.info(f"Eliminando endpoint: {endpoint}")
            try:
                sagemaker.delete_endpoint(EndpointName=endpoint)
                waiter = sagemaker.get_waiter('endpoint_deleted')
                waiter.wait(EndpointName=endpoint)
                logger.info(f"Endpoint {endpoint} eliminado correctamente")
            except Exception as e:
                logger.error(f"Error eliminando endpoint {endpoint}: {str(e)}")
        
        # Eliminar todas las configuraciones
        for config_name in resources['endpoint_configs']:
            logger.info(f"Eliminando configuración: {config_name}")
            try:
                sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
                logger.info(f"Configuración {config_name} eliminada correctamente")
            except Exception as e:
                logger.error(f"Error eliminando configuración {config_name}: {str(e)}")
        
        # Eliminar todos los modelos
        for model_name in resources['models']:
            logger.info(f"Eliminando modelo: {model_name}")
            try:
                sagemaker.delete_model(ModelName=model_name)
                logger.info(f"Modelo {model_name} eliminado correctamente")
            except Exception as e:
                logger.error(f"Error eliminando modelo {model_name}: {str(e)}")
        
        # Esperar a que todos los recursos se liberen
        logger.info("Esperando a que AWS libere los recursos...")
        time.sleep(60)  # Esperar 1 minuto para asegurar que los recursos se liberen

        # Nombres de recursos
        model_name = config['nombre']
        endpoint_config_name = f"{model_name}-config"
        endpoint_name = f"{model_name}-endpoint"

        # Crear modelo
        logger.info(f"Creando modelo {model_name}...")
        
        # Usar imagen oficial de AWS para SageMaker
        account_id = "763104351884"  # AWS Deep Learning Container account
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
        
        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                'Image': image_uri,
                'Environment': {
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': region,
                    'MODEL_LOADING_TIMEOUT': '3600',
                    'INFERENCE_TIMEOUT': '3600',
                    'SERVING_MODE': 'PYTORCH',
                    'PYTORCH_JIT': 'false',
                    'MMS_DEFAULT_RESPONSE_TIMEOUT': '3600',
                    'MAX_REQUEST_SIZE': '10485760',
                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'
                }
            }
        )
        
        # Crear configuración del endpoint
        logger.info(f"Creando configuración del endpoint...")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': config['instancia'],
                    'InitialInstanceCount': 1
                }
            ]
        )
        
        # Crear endpoint
        logger.info(f"Creando endpoint {endpoint_name}...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        # Esperar a que el endpoint esté listo
        logger.info("Esperando a que el endpoint esté listo...")
        waiter = sagemaker.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        logger.info(f"¡Endpoint {endpoint_name} creado y listo para usar!")
        
        # Guardar estado del endpoint
        save_endpoint_state(endpoint_name, modelo_elegido)
        
        return endpoint_name
        
    except Exception as e:
        logger.error(f"Error creando endpoint: {str(e)}")
        if sagemaker and model_name:
            logger.info("Limpiando recursos debido al error...")
            cleanup_resources(sagemaker, model_name)
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
    parser = argparse.ArgumentParser(description='Configurar endpoint de SageMaker para Llama 2')
    parser.add_argument('--modelo', choices=['7b', '13b', '70b'], default='13b',
                      help='Versión del modelo a usar (default: 13b)')
    parser.add_argument('--force', action='store_true',
                      help='Forzar recreación del endpoint aunque exista')
    parser.add_argument('--cleanup', action='store_true',
                      help='Eliminar todos los recursos al terminar')
    
    args = parser.parse_args()
    
    try:
        endpoint_name = create_sagemaker_endpoint(args.modelo, args.force)
        if args.cleanup:
            logger.info("Limpiando recursos...")
            region = os.environ.get('AWS_REGION', 'us-east-1')
            session = boto3.Session(region_name=region)
            sagemaker = session.client('sagemaker')
            cleanup_resources(sagemaker, MODELO_CONFIG[args.modelo]['nombre'])
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1) 