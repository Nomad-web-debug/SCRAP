provider "aws" {
  region = "us-east-2"
}

# Proveedor random
provider "random" {
}

# Datos de VPC existente (si existe)
data "aws_vpcs" "existing" {
  tags = {
    Name = "clasificador-vpc"
  }
}

locals {
  use_existing_vpc = length(data.aws_vpcs.existing.ids) > 0
  vpc_id = local.use_existing_vpc ? data.aws_vpcs.existing.ids[0] : aws_vpc.main[0].id
}

# VPC y Networking
resource "aws_vpc" "main" {
  count = local.use_existing_vpc ? 0 : 1
  
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Name = "clasificador-vpc"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  count = local.use_existing_vpc ? 0 : 1
  vpc_id = local.vpc_id

  tags = {
    Name = "clasificador-igw"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Subnet pública
data "aws_subnets" "existing" {
  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
  
  tags = {
    Name = "clasificador-subnet-public"
  }
}

locals {
  use_existing_subnet = length(data.aws_subnets.existing.ids) > 0
  subnet_id = local.use_existing_subnet ? data.aws_subnets.existing.ids[0] : aws_subnet.public[0].id
}

resource "aws_subnet" "public" {
  count = local.use_existing_subnet ? 0 : 1
  
  vpc_id     = local.vpc_id
  cidr_block = "10.0.1.0/24"
  availability_zone = "us-east-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "clasificador-subnet-public"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Route Table
resource "aws_route_table" "public" {
  count = local.use_existing_vpc ? 0 : 1
  vpc_id = local.vpc_id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = {
    Name = "clasificador-rt-public"
  }
}

resource "aws_route_table_association" "public" {
  count = local.use_existing_vpc ? 0 : 1
  subnet_id      = local.subnet_id
  route_table_id = aws_route_table.public[0].id
}

# Security Group
resource "aws_security_group" "ec2" {
  name_prefix = "clasificador-sg-${random_string.suffix.result}"
  description = "Security group for clasificador EC2"
  vpc_id      = local.vpc_id

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

  lifecycle {
    create_before_destroy = true
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
  lower   = true
  min_lower = 2
  min_numeric = 2
  numeric = true
}

# S3 Bucket
resource "aws_s3_bucket" "app" {
  bucket = "clasificador-docs-${random_string.suffix.result}"
  force_destroy = true

  tags = {
    Name = "clasificador-documentos"
    Environment = "production"
    ManagedBy = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "app" {
  bucket = aws_s3_bucket.app.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "app" {
  bucket = aws_s3_bucket.app.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
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
  name_prefix = "clasificador-documentos-template-"
  image_id = "ami-0430580de6244e02e"
  instance_type = "t3.medium"

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups = [aws_security_group.ec2.id]
    delete_on_termination = true
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

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "clasificador-launch-template"
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {
  name_prefix = "clasificador-documentos-asg-"
  desired_capacity = 0
  max_size = 1
  min_size = 0
  vpc_zone_identifier = [local.subnet_id]
  health_check_type = "EC2"
  health_check_grace_period = 300

  launch_template {
    id = aws_launch_template.app.id
    version = "$Latest"
  }

  tag {
    key = "Name"
    value = "ClasificadorDocumentos"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes = [desired_capacity]
  }

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }
}

# CloudWatch Alarm para Auto Scaling
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name = "clasificador-documentos-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods = "2"
  metric_name = "CPUUtilization"
  namespace = "AWS/EC2"
  period = "300"
  statistic = "Average"
  threshold = "70"
  alarm_description = "Scale up if CPU > 70% for 10 minutes"
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }

  alarm_actions = [aws_autoscaling_policy.scale_up.arn]
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name = "clasificador-documentos-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods = "2"
  metric_name = "CPUUtilization"
  namespace = "AWS/EC2"
  period = "300"
  statistic = "Average"
  threshold = "30"
  alarm_description = "Scale down if CPU < 30% for 10 minutes"
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }

  alarm_actions = [aws_autoscaling_policy.scale_down.arn]
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  name = "clasificador-scale-up"
  scaling_adjustment = 1
  adjustment_type = "ChangeInCapacity"
  cooldown = 300
  autoscaling_group_name = aws_autoscaling_group.app.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name = "clasificador-scale-down"
  scaling_adjustment = -1
  adjustment_type = "ChangeInCapacity"
  cooldown = 300
  autoscaling_group_name = aws_autoscaling_group.app.name
}

# Outputs
output "vpc_id" {
  value = local.vpc_id
}

output "subnet_id" {
  value = local.subnet_id
}

output "security_group_id" {
  value = aws_security_group.ec2.id
}

output "s3_bucket" {
  value       = aws_s3_bucket.app.id
  description = "Nombre del bucket S3"
  depends_on  = [aws_s3_bucket.app]
} 