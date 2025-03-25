import boto3
import time

# Configurar la región
region = 'us-east-2'

def wait_for_function_update(lambda_client, function_name, max_attempts=10):
    """Espera hasta que la función Lambda termine de actualizarse"""
    for i in range(max_attempts):
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            if response['Configuration']['State'] == 'Active':
                return True
            print(f"Esperando que la función termine de actualizarse... intento {i+1}/{max_attempts}")
            time.sleep(10)
        except Exception as e:
            print(f"Error al verificar estado de la función: {str(e)}")
            return False
    return False

def delete_all_resources():
    """Elimina todos los recursos AWS relacionados con el proyecto"""
    
    print("Iniciando limpieza de recursos...")
    
    # Inicializar clientes con región específica
    lambda_client = boto3.client('lambda', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    # 1. Eliminar buckets S3 (primero porque otros servicios pueden depender de ellos)
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
                            print(f"  - Eliminado objeto: {obj['Key']}")
                    
                    # Luego eliminar el bucket
                    s3.delete_bucket(Bucket=bucket_name)
                    print(f"✓ Bucket {bucket_name} eliminado")
                    time.sleep(2)
                except Exception as e:
                    print(f"No se pudo eliminar el bucket {bucket_name}: {str(e)}")
    except Exception as e:
        print(f"Error al listar/eliminar buckets: {str(e)}")

    # 2. Eliminar tablas DynamoDB
    print("\nEliminando tablas DynamoDB...")
    try:
        tables = dynamodb.list_tables()['TableNames']
        for table in tables:
            if any(x in table.lower() for x in ['documentos', 'metadata', 'training', 'chat']):
                try:
                    dynamodb.delete_table(TableName=table)
                    print(f"✓ Tabla {table} eliminada")
                    time.sleep(3)
                except Exception as e:
                    print(f"No se pudo eliminar la tabla {table}: {str(e)}")
    except Exception as e:
        print(f"Error al listar tablas: {str(e)}")

    # 3. Eliminar capas Lambda
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
                try:
                    lambda_client.delete_layer_version(
                        LayerName=layer_name,
                        VersionNumber=version['Version']
                    )
                    print(f"✓ Capa {layer_name} versión {version['Version']} eliminada")
                    time.sleep(1)
                except Exception as e:
                    print(f"No se pudo eliminar la versión {version['Version']} de {layer_name}: {str(e)}")
        except Exception as e:
            print(f"No se pudo listar versiones de {layer_name}: {str(e)}")

    # 4. Esperar un momento antes de intentar eliminar la función Lambda
    print("\nEsperando 30 segundos antes de eliminar la función Lambda...")
    time.sleep(30)

    # 5. Intentar eliminar la función Lambda
    print("\nEliminando función Lambda...")
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            lambda_client.delete_function(FunctionName='ClasificadorDocumentosIA')
            print("✓ Función Lambda eliminada")
            break
        except Exception as e:
            print(f"Intento {attempt + 1}/{max_attempts}: No se pudo eliminar la función Lambda: {str(e)}")
            if attempt < max_attempts - 1:
                print("Esperando 20 segundos antes del siguiente intento...")
                time.sleep(20)

    print("\nLimpieza completada.")

if __name__ == "__main__":
    delete_all_resources() 