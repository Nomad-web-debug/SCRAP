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

# VPC y Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "clasificador-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "clasificador-subnet-public"
  }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-2b"

  tags = {
    Name = "clasificador-subnet-private"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "clasificador-igw"
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "clasificador-rt-public"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Group para EC2
resource "aws_security_group" "ec2" {
  name        = "clasificador-sg-ec2"
  description = "Security group for EC2 instance"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "clasificador-sg-ec2"
  }
}

# S3 Bucket
resource "aws_s3_bucket" "docs" {
  bucket = "clasificador-docs-${random_string.suffix.result}"
}

resource "aws_s3_bucket_lifecycle_configuration" "docs_lifecycle" {
  bucket = aws_s3_bucket.docs.id

  rule {
    id     = "cleanup"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 60
      storage_class = "GLACIER"
    }

    expiration {
      days = 90
    }
  }
}

# RDS Instance
resource "aws_db_subnet_group" "main" {
  name       = "clasificador-db-subnet"
  subnet_ids = [aws_subnet.private.id, aws_subnet.public.id]
}

resource "aws_security_group" "rds" {
  name        = "clasificador-sg-rds"
  description = "Security group for RDS instance"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id]
  }

  tags = {
    Name = "clasificador-sg-rds"
  }
}

resource "aws_db_instance" "main" {
  identifier        = "clasificador-db"
  engine            = "postgres"
  engine_version    = "13.7"
  instance_class    = "db.t3.micro"
  allocated_storage = 20

  db_name  = "clasificador"
  username = "dbadmin"
  password = random_password.db_password.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  skip_final_snapshot = true
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  tags = {
    Name = "clasificador-db"
  }
}

# EC2 Launch Template
resource "aws_launch_template" "main" {
  name_prefix   = "clasificador-lt"
  image_id      = "ami-0430580de6244e02e" # Amazon Linux 2 AMI
  instance_type = "t3.micro"

  network_interfaces {
    associate_public_ip_address = true
    security_groups            = [aws_security_group.ec2.id]
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              yum update -y
              yum install -y python3-pip git
              pip3 install boto3 pandas numpy torch transformers
              EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "clasificador-ec2"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "main" {
  name                = "clasificador-asg"
  desired_capacity    = 0
  max_size           = 1
  min_size           = 0
  target_group_arns  = [aws_lb_target_group.main.arn]
  vpc_zone_identifier = [aws_subnet.public.id]

  launch_template {
    id      = aws_launch_template.main.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value              = "clasificador-asg"
    propagate_at_launch = true
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "clasificador-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ec2.id]
  subnets           = [aws_subnet.public.id, aws_subnet.private.id]
}

resource "aws_lb_target_group" "main" {
  name     = "clasificador-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
}

resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}

# Random string para nombres únicos
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Contraseña aleatoria para RDS
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Outputs
output "rds_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "s3_bucket" {
  value = aws_s3_bucket.docs.bucket
}

output "alb_dns" {
  value = aws_lb.main.dns_name
} 