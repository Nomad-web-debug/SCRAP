import boto3

# Configurar la región
region = 'us-east-2'

def delete_all_resources():
    """Elimina todos los recursos AWS relacionados con el proyecto"""
    
    print("Iniciando limpieza de recursos...")
    
    # Inicializar clientes con región específica
    lambda_client = boto3.client('lambda', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    # 1. Eliminar función Lambda
    print("\nEliminando función Lambda...")
    try:
        lambda_client.delete_function(FunctionName='ClasificadorDocumentosIA')
        print("✓ Función Lambda eliminada")
    except Exception as e:
        print(f"No se pudo eliminar la función Lambda: {str(e)}")

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
            print(f"No se pudo eliminar la capa {layer_name}: {str(e)}")

    # 3. Eliminar tablas DynamoDB
    print("\nEliminando tablas DynamoDB...")
    try:
        tables = dynamodb.list_tables()['TableNames']
        for table in tables:
            if any(x in table.lower() for x in ['documentos', 'metadata', 'training', 'chat']):
                try:
                    dynamodb.delete_table(TableName=table)
                    print(f"✓ Tabla {table} eliminada")
                except Exception as e:
                    print(f"No se pudo eliminar la tabla {table}: {str(e)}")
    except Exception as e:
        print(f"Error al listar tablas: {str(e)}")

    # 4. Eliminar buckets S3
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
                except Exception as e:
                    print(f"No se pudo eliminar el bucket {bucket_name}: {str(e)}")
    except Exception as e:
        print(f"Error al listar/eliminar buckets: {str(e)}")

    print("\nLimpieza completada.")

if __name__ == "__main__":
    delete_all_resources() 