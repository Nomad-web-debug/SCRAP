provider "aws" {
  region = "us-east-2"
}

# Proveedor random
provider "random" {
}

# VPC y Networking
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Name = "clasificador-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "clasificador-igw"
  }
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "us-east-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "clasificador-subnet-public"
  }
}

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

# Security Group
resource "aws_security_group" "ec2" {
  name        = "clasificador-sg"
  description = "Security group for clasificador EC2"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
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
    Name = "clasificador-sg"
  }
}

# Generador de sufijo aleatorio
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket
resource "aws_s3_bucket" "app" {
  bucket = "clasificador-documentos-${random_string.suffix.result}"
}

resource "aws_s3_bucket_versioning" "app" {
  bucket = aws_s3_bucket.app.id
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM Role para EC2
resource "aws_iam_role" "ec2_role" {
  name = "clasificador_ec2_role_${random_string.suffix.result}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "clasificador-ec2-role"
  }
}

resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "clasificador_ec2_profile_${random_string.suffix.result}"
  role = aws_iam_role.ec2_role.name
}

# Launch Template
resource "aws_launch_template" "app" {
  name = "clasificador-documentos-template-${random_string.suffix.result}"
  image_id = "ami-0430580de6244e02e"
  instance_type = "t3.medium"

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups = [aws_security_group.ec2.id]
    subnet_id = aws_subnet.public.id
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              yum update -y
              yum install -y python3-pip unzip
              mkdir -p /opt/clasificador
              cd /opt/clasificador
              aws s3 cp s3://${aws_s3_bucket.app.bucket}/app.zip .
              unzip app.zip
              pip3 install -r requirements.txt
              chown -R ec2-user:ec2-user /opt/clasificador
              su -c "python3 app/main.py" ec2-user
              EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "ClasificadorDocumentos"
    }
  }

  tags = {
    Name = "clasificador-launch-template"
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {
  name = "clasificador-documentos-asg-${random_string.suffix.result}"
  desired_capacity = 0
  max_size = 1
  min_size = 0
  vpc_zone_identifier = [aws_subnet.public.id]

  launch_template {
    id = aws_launch_template.app.id
    version = "$Latest"
  }

  tag {
    key = "Name"
    value = "ClasificadorDocumentos"
    propagate_at_launch = true
  }
}

# CloudWatch Alarm
resource "aws_cloudwatch_metric_alarm" "cpu" {
  alarm_name = "clasificador-documentos-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods = "2"
  metric_name = "CPUUtilization"
  namespace = "AWS/EC2"
  period = "300"
  statistic = "Average"
  threshold = "70"
  alarm_description = "This metric monitors EC2 CPU utilization"
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }
}

# Outputs
output "vpc_id" {
  value = aws_vpc.main.id
}

output "subnet_id" {
  value = aws_subnet.public.id
}

output "security_group_id" {
  value = aws_security_group.ec2.id
}

output "s3_bucket" {
  value = aws_s3_bucket.app.bucket
} 