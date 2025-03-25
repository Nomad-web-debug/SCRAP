import boto3

# Configurar la región
region = 'us-east-2'

def check_resources():
    """Verifica qué recursos AWS relacionados con el proyecto existen"""
    
    print("Verificando recursos existentes...")
    
    # Inicializar clientes
    lambda_client = boto3.client('lambda', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    # 1. Verificar función Lambda
    print("\nFunción Lambda:")
    try:
        lambda_client.get_function(FunctionName='ClasificadorDocumentosIA')
        print("✗ La función ClasificadorDocumentosIA existe")
    except Exception as e:
        print("✓ No existe función Lambda")

    # 2. Verificar capas Lambda
    print("\nCapas Lambda:")
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
            if versions['LayerVersions']:
                print(f"✗ La capa {layer_name} existe con {len(versions['LayerVersions'])} versiones")
        except Exception:
            print(f"✓ No existe la capa {layer_name}")

    # 3. Verificar buckets S3
    print("\nBuckets S3:")
    try:
        buckets = s3.list_buckets()['Buckets']
        found = False
        for bucket in buckets:
            if any(x in bucket['Name'].lower() for x in ['clasificador', 'docs', 'raw', 'processed']):
                print(f"✗ El bucket {bucket['Name']} existe")
                found = True
        if not found:
            print("✓ No existen buckets relacionados")
    except Exception as e:
        print(f"Error al listar buckets: {str(e)}")

    # 4. Verificar tablas DynamoDB
    print("\nTablas DynamoDB:")
    try:
        tables = dynamodb.list_tables()['TableNames']
        found = False
        for table in tables:
            if any(x in table.lower() for x in ['documentos', 'metadata', 'training', 'chat']):
                print(f"✗ La tabla {table} existe")
                found = True
        if not found:
            print("✓ No existen tablas relacionadas")
    except Exception as e:
        print(f"Error al listar tablas: {str(e)}")

if __name__ == "__main__":
    check_resources() 