provider "aws" {
  region = "us-east-2"
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

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "us-east-2a"

  tags = {
    Name = "clasificador-subnet-public"
  }
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

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# S3 Bucket
resource "aws_s3_bucket" "app" {
  bucket = "clasificador-documentos-app"
}

resource "aws_s3_bucket_versioning" "app" {
  bucket = aws_s3_bucket.app.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Launch Template
resource "aws_launch_template" "app" {
  name = "clasificador-documentos-template"
  image_id = "ami-0430580de6244e02e"
  instance_type = "t3.medium"

  network_interfaces {
    associate_public_ip_address = true
    security_groups = [aws_security_group.ec2.id]
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              cd /opt/clasificador
              aws s3 cp s3://clasificador-documentos-app/app.zip .
              unzip app.zip
              pip3 install -r requirements.txt
              chown -R ec2-user:ec2-user /opt/clasificador
              su -c "python3 app/main.py" ec2-user
              EOF
  )
}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {
  name = "clasificador-documentos-asg"
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