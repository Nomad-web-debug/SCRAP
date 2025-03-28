terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

# Bucket S3 fijo para almacenar documentos
resource "aws_s3_bucket" "documentos_bucket" {
  bucket = "clasificador-docs-principal"
  force_destroy = false
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

  rule {
    id     = "archivos_temporales"
    status = "Enabled"

    expiration {
      days = 30  # Los archivos temporales se eliminan después de 30 días
    }
  }
}

# Política de acceso al bucket
resource "aws_s3_bucket_policy" "bucket_policy" {
  bucket = aws_s3_bucket.documentos_bucket.id

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