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
            
            # Crear el rol
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Adjuntar políticas necesarias
            attach_required_policies(iam_client, role_name)
            
            logger.info(f"Rol {role_name} creado exitosamente")
            return response['Role']['Arn']
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
    
    # Política inline para acceder a la imagen específica de Llama 2
    ecr_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:GetAuthorizationToken",
                    "ecr:DescribeImages",
                    "ecr:ListImages",
                    "ecr:BatchCheckLayerAvailability"
                ],
                "Resource": [
                    "arn:aws:ecr:us-east-1:456233644234:repository/jumpstart-inference-meta-textgeneration-llama-2-70b",
                    "arn:aws:ecr:us-east-1:763104351884:repository/djl-inference"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken"
                ],
                "Resource": "*"
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
    
    # Adjuntar política inline para ECR
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName='ECRAccessForLlama2',
            PolicyDocument=json.dumps(ecr_policy)
        )
        logger.info("Política inline para ECR adjuntada al rol")
    except Exception as e:
        logger.error(f"Error adjuntando política inline: {str(e)}")
        raise

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
        iam = session.client('iam')
        sts = session.client('sts')
        
        logger.info(f"Configurando servicios en la región: {region}")
        
        # Obtener ID de cuenta
        account_id = sts.get_caller_identity()["Account"]
        logger.info(f"ID de cuenta AWS: {account_id}")
        
        # Obtener o crear rol de ejecución
        role_arn = get_or_create_role(iam)
        logger.info(f"Usando rol: {role_arn}")
        
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
            
            # Usar la imagen oficial de SageMaker para Llama 2
            image_uri = f"456233644234.dkr.ecr.{region}.amazonaws.com/jumpstart-inference-meta-textgeneration-llama-2-70b:1.0.0"
            
            sagemaker.create_model(
                ModelName=model_name,
                ExecutionRoleArn=role_arn,
                PrimaryContainer={
                    'Image': image_uri,
                    'Environment': {
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': region,
                        'MAX_INPUT_LENGTH': '2048',
                        'MAX_TOTAL_TOKENS': '4096'
                    }
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
                    'VariantName': 'AllTraffic'
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
            logger.info(f"Esperando a que el endpoint esté listo... Estado actual: {status}")
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
    try:
        endpoint_name = create_sagemaker_endpoint()
        if test_endpoint(endpoint_name):
            print(f"Endpoint {endpoint_name} configurado y funcionando correctamente")
        else:
            print("Error al probar el endpoint")
            exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 