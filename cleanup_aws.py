import boto3
import os
import time

# Configurar credenciales desde variables de entorno
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
region = 'us-east-2'

def wait_for_function_update(lambda_client, function_name):
    print(f"Esperando a que la función {function_name} termine de actualizarse...")
    while True:
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            if response['Configuration']['LastUpdateStatus'] == 'Successful':
                break
            time.sleep(10)
        except lambda_client.exceptions.ResourceNotFoundException:
            break
        except Exception as e:
            print(f"Error al verificar estado: {str(e)}")
            break

def delete_all_resources():
    """Elimina todos los recursos AWS relacionados con el proyecto"""
    
    print("Iniciando limpieza de recursos...")
    
    # Inicializar clientes con credenciales
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=region
    )
    
    lambda_client = session.client('lambda')
    s3 = session.client('s3')
    dynamodb = session.client('dynamodb')
    iam = session.client('iam')
    
    # 1. Eliminar función Lambda y su rol IAM
    print("\nEliminando función Lambda y rol IAM...")
    try:
        # Esperar a que termine la actualización
        wait_for_function_update(lambda_client, 'ClasificadorDocumentosIA')
        
        # Obtener el rol de la función Lambda
        lambda_info = lambda_client.get_function(FunctionName='ClasificadorDocumentosIA')
        role_arn = lambda_info['Configuration']['Role']
        role_name = role_arn.split('/')[-1]
        
        # Eliminar la función Lambda
        lambda_client.delete_function(FunctionName='ClasificadorDocumentosIA')
        print("✓ Función Lambda eliminada")
        
        # Eliminar políticas adjuntas al rol
        try:
            attached_policies = iam.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
            for policy in attached_policies:
                iam.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy['PolicyArn']
                )
            # Eliminar el rol
            iam.delete_role(RoleName=role_name)
            print(f"✓ Rol IAM {role_name} eliminado")
        except Exception as e:
            print(f"Error al eliminar rol IAM: {str(e)}")
            
    except Exception as e:
        print(f"Error con la función Lambda: {str(e)}")

    # 2. Eliminar capas Lambda
    print("\nEliminando capas Lambda...")
    layers = [
        "ClasificadorBase",
        "ClasificadorProcessing",
        "ClasificadorML",
        "ClasificadorTorch",
        "ClasificadorNLP",
        "ClasificadorData"
    ]
    for layer_name in layers:
        try:
            versions = lambda_client.list_layer_versions(LayerName=layer_name)
            for version in versions['LayerVersions']:
                lambda_client.delete_layer_version(
                    LayerName=layer_name,
                    VersionNumber=version['Version']
                )
            print(f"✓ Capa {layer_name} eliminada")
        except Exception as e:
            print(f"Error con capa {layer_name}: {str(e)}")

    # 3. Eliminar buckets S3
    print("\nEliminando buckets S3...")
    try:
        buckets = s3.list_buckets()['Buckets']
        for bucket in buckets:
            bucket_name = bucket['Name']
            if any(x in bucket_name.lower() for x in ['clasificador', 'docs', 'raw', 'processed']):
                try:
                    # Primero vaciar el bucket
                    objects = s3.list_objects_v2(Bucket=bucket_name)
                    if 'Contents' in objects:
                        for obj in objects['Contents']:
                            s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                    # Luego eliminar el bucket
                    s3.delete_bucket(Bucket=bucket_name)
                    print(f"✓ Bucket {bucket_name} eliminado")
                except Exception as e:
                    print(f"Error al eliminar bucket {bucket_name}: {str(e)}")
    except Exception as e:
        print(f"Error al listar buckets: {str(e)}")

    # 4. Eliminar tablas DynamoDB
    print("\nEliminando tablas DynamoDB...")
    try:
        tables = dynamodb.list_tables()['TableNames']
        for table in tables:
            if any(x in table.lower() for x in ['documentos', 'metadata', 'training', 'chat']):
                try:
                    dynamodb.delete_table(TableName=table)
                    print(f"✓ Tabla {table} eliminada")
                except Exception as e:
                    print(f"Error al eliminar tabla {table}: {str(e)}")
    except Exception as e:
        print(f"Error al listar tablas: {str(e)}")

    print("\nLimpieza completada.")

if __name__ == "__main__":
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("Error: Las credenciales de AWS no están configuradas")
        exit(1)
    delete_all_resources() 