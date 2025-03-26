#!/bin/bash

# Verificar si hay archivos en el directorio de entrada
FILES_COUNT=$(ls -1 /opt/clasificador/documents/input/*.pdf 2>/dev/null | wc -l)

# Verificar uso de CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

# Si no hay archivos y el CPU está bajo, apagar la instancia
if [ "$FILES_COUNT" -eq 0 ] && [ $(echo "$CPU_USAGE < 10" | bc) -eq 1 ]; then
    # Obtener ID de instancia
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    
    # Apagar instancia usando AWS CLI
    aws ec2 stop-instances --instance-ids $INSTANCE_ID --region us-east-2
fi 