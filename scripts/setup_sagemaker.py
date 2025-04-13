import boto3
import json
import time
import logging
import os
from botocore.exceptions import ClientError
import argparse
from datetime import datetime
from typing import Dict, Tuple

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
        'instancia': 'ml.g5.xlarge',
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
    
    # Asegurar que el directorio existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar en el directorio data
    file_path = os.path.join('data', 'endpoint_state.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info(f"Estado del endpoint guardado en {file_path}")
    
    # Asegurar que el archivo se suba al repositorio
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        print(f"::set-output name=endpoint_state_file::{file_path}")

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

def check_service_limits(sagemaker):
    """
    Verifica los límites de servicio y endpoints activos
    """
    try:
        # Listar todos los endpoints activos
        response = sagemaker.list_endpoints()
        active_endpoints = response['Endpoints']
        
        logger.info(f"Endpoints activos encontrados: {len(active_endpoints)}")
        for endpoint in active_endpoints:
            logger.info(f"- {endpoint['EndpointName']} (Estado: {endpoint['EndpointStatus']})")
            
            # Obtener detalles del endpoint
            endpoint_details = sagemaker.describe_endpoint(EndpointName=endpoint['EndpointName'])
            config_name = endpoint_details['EndpointConfigName']
            
            # Obtener detalles de la configuración
            config_details = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
            instance_type = config_details['ProductionVariants'][0]['InstanceType']
            
            logger.info(f"  Tipo de instancia: {instance_type}")
            logger.info(f"  Configuración: {config_name}")
        
        # Listar todos los modelos
        response = sagemaker.list_models()
        active_models = response['Models']
        
        logger.info(f"\nModelos activos encontrados: {len(active_models)}")
        for model in active_models:
            logger.info(f"- {model['ModelName']}")
        
        return active_endpoints, active_models
        
    except Exception as e:
        logger.error(f"Error verificando límites de servicio: {str(e)}")
        return [], []

def create_sagemaker_endpoint(config: Dict[str, Any], region: str) -> Tuple[str, str]:
    """
    Crea un endpoint de SageMaker para el modelo especificado
    
    Args:
        config (Dict[str, Any]): Configuración del modelo y endpoint
        region (str): Región de AWS
        
    Returns:
        Tuple[str, str]: Nombre del endpoint y ARN del rol
    """
    try:
        sagemaker = boto3.client('sagemaker', region_name=region)
        
        # Obtener ARN del rol
        role_arn = get_or_create_role(boto3.client('iam'))
        
        # Crear nombre único para el modelo y endpoint
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        model_name = f"{config['nombre_modelo']}-{timestamp}"
        endpoint_config_name = f"{model_name}-config"
        endpoint_name = f"{model_name}-endpoint"
        
        # Obtener URI de la imagen del container
        image_uri = get_container_image(region)
        
        logger.info(f"Creando modelo SageMaker: {model_name}")
        logger.info(f"Usando imagen: {image_uri}")
        logger.info(f"Rol ARN: {role_arn}")
        
        # Crear el modelo en SageMaker con la configuración necesaria
        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                'Image': image_uri,
                'Environment': {
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': region,
                    'HF_MODEL_ID': config['modelo_hf'],  # Variable requerida por containers JumpStart
                    'MODEL_LOADING_TIMEOUT': '3600',
                    'INFERENCE_TIMEOUT': '3600',
                    'MAX_INPUT_LENGTH': '2048',
                    'MAX_TOTAL_TOKENS': '4096',
                    'HF_TASK': 'text-generation'
                }
            }
        )
        
        logger.info(f"Modelo creado: {model_name}")
        
        # Crear configuración del endpoint
        logger.info(f"Creando configuración del endpoint: {endpoint_config_name}")
        
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': config['tipo_instancia'],
                    'InitialInstanceCount': config['num_instancias']
                }
            ]
        )
        
        logger.info(f"Configuración del endpoint creada: {endpoint_config_name}")
        
        # Crear el endpoint
        logger.info(f"Creando endpoint: {endpoint_name}")
        
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        logger.info(f"Endpoint creado: {endpoint_name}")
        logger.info("Esperando a que el endpoint esté en servicio...")
        
        # Esperar a que el endpoint esté listo
        waiter = sagemaker.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
        )
        
        logger.info("¡Endpoint listo para usar!")
        
        return endpoint_name, role_arn
        
    except Exception as e:
        logger.error("=== Error al crear el endpoint ===")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Mensaje: {str(e)}")
        logger.error(f"Región: {region}")
        logger.error(f"Configuración: {json.dumps(config, indent=2)}")
        logger.error("=== Fin del error ===")
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
        endpoint_name, role_arn = create_sagemaker_endpoint(MODELO_CONFIG[args.modelo], os.environ.get('AWS_REGION', 'us-east-1'))
        if args.cleanup:
            logger.info("Limpiando recursos...")
            region = os.environ.get('AWS_REGION', 'us-east-1')
            session = boto3.Session(region_name=region)
            sagemaker = session.client('sagemaker')
            cleanup_resources(sagemaker, MODELO_CONFIG[args.modelo]['nombre'])
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1) 