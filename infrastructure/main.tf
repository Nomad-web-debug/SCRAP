terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Configurar el backend para mantener el estado en S3
  backend "s3" {
    bucket = "terraform-state-scraper-ia"
    key    = "prod/terraform.tfstate"
    region = "us-east-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# Bucket S3 fijo para almacenar documentos
resource "aws_s3_bucket" "documentos_bucket" {
  bucket = "clasificador-docs-principal"
  force_destroy = false
}

# Configurar acceso público al bucket
resource "aws_s3_bucket_public_access_block" "documentos_access" {
  bucket = aws_s3_bucket.documentos_bucket.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# Configurar el bucket
resource "aws_s3_bucket_versioning" "bucket_versioning" {
  bucket = aws_s3_bucket.documentos_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Configurar el lifecycle del bucket
resource "aws_s3_bucket_lifecycle_configuration" "bucket_lifecycle" {
  bucket = aws_s3_bucket.documentos_bucket.id
  depends_on = [aws_s3_bucket_versioning.bucket_versioning]

  rule {
    id     = "archivos_temporales"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 30
    }
  }
}

# Política de acceso al bucket
resource "aws_s3_bucket_policy" "bucket_policy" {
  bucket = aws_s3_bucket.documentos_bucket.id
  depends_on = [aws_s3_bucket_public_access_block.documentos_access]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowReadWrite"
        Effect    = "Allow"
        Principal = "*"
        Action    = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.documentos_bucket.arn,
          "${aws_s3_bucket.documentos_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Habilitar CORS para permitir acceso desde cualquier origen
resource "aws_s3_bucket_cors_configuration" "bucket_cors" {
  bucket = aws_s3_bucket.documentos_bucket.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# S3 Buckets
resource "aws_s3_bucket" "raw" {
  bucket = "${var.project_prefix}-raw"
}

resource "aws_s3_bucket" "processed" {
  bucket = "${var.project_prefix}-processed"
}

resource "aws_s3_bucket" "metadata" {
  bucket = "${var.project_prefix}-metadata"
}

# DynamoDB
resource "aws_dynamodb_table" "legal_docs" {
  name           = "${var.project_prefix}-documents"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"
  stream_enabled = true

  attribute {
    name = "id"
    type = "S"
  }

  attribute {
    name = "documento"
    type = "S"
  }

  attribute {
    name = "rama_derecho"
    type = "S"
  }

  global_secondary_index {
    name               = "DocumentoIndex"
    hash_key           = "documento"
    projection_type    = "ALL"
  }

  global_secondary_index {
    name               = "RamaDerechoIndex"
    hash_key           = "rama_derecho"
    projection_type    = "ALL"
  }
}

# IAM Role para SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Role para Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Lambda Functions
resource "aws_lambda_function" "preprocessor" {
  filename         = "lambda/preprocessor.zip"
  function_name    = "${var.project_prefix}-preprocessor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.10"
  timeout         = 300
  memory_size     = 2048

  environment {
    variables = {
      RAW_BUCKET = aws_s3_bucket.raw.id
      PROCESSED_BUCKET = aws_s3_bucket.processed.id
      DYNAMODB_TABLE = aws_dynamodb_table.legal_docs.name
    }
  }
}

resource "aws_lambda_function" "processor" {
  filename         = "lambda/processor.zip"
  function_name    = "${var.project_prefix}-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "python3.10"
  timeout         = 900
  memory_size     = 10240

  environment {
    variables = {
      SAGEMAKER_ENDPOINT = aws_sagemaker_endpoint.llama.name
      PROCESSED_BUCKET = aws_s3_bucket.processed.id
      METADATA_BUCKET = aws_s3_bucket.metadata.id
      DYNAMODB_TABLE = aws_dynamodb_table.legal_docs.name
    }
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "llama" {
  name = "${var.project_prefix}-llama-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name            = aws_sagemaker_model.llama.name
    initial_instance_count = 1
    instance_type         = "ml.g5.12xlarge"
    
    serverless_config {
      max_concurrency = 20
      memory_size_in_mb = 16384
    }
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "llama" {
  name                 = "${var.project_prefix}-llama-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.llama.name
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "cost_alarm" {
  alarm_name          = "${var.project_prefix}-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period             = "21600" # 6 horas
  statistic          = "Maximum"
  threshold          = var.cost_threshold
  alarm_description  = "Alerta cuando los costos estimados superan el umbral"
  alarm_actions      = [aws_sns_topic.alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}

# SNS Topic para alertas
resource "aws_sns_topic" "alerts" {
  name = "${var.project_prefix}-alerts"
}

# Variables
variable "aws_region" {
  description = "AWS Region"
  default     = "us-east-1"
}

variable "project_prefix" {
  description = "Prefix for all resources"
  default     = "legal-docs"
}

variable "cost_threshold" {
  description = "Umbral de costos diarios en USD"
  default     = 100
}

# Outputs
output "sagemaker_endpoint" {
  value = aws_sagemaker_endpoint.llama.name
}

output "preprocessor_lambda" {
  value = aws_lambda_function.preprocessor.function_name
}

output "processor_lambda" {
  value = aws_lambda_function.processor.function_name
}

output "raw_bucket" {
  value = aws_s3_bucket.raw.id
}

output "processed_bucket" {
  value = aws_s3_bucket.processed.id
}

output "dynamodb_table" {
  value = aws_dynamodb_table.legal_docs.name
} 