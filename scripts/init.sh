#!/bin/bash

# Actualizar sistema
yum update -y
yum install -y python3-pip git postgresql-devel python3-devel gcc

# Crear directorios
mkdir -p /opt/clasificador/documents/{input,processed}

# Clonar repositorio
cd /opt/clasificador
git clone https://github.com/TU_USUARIO/TU_REPO.git .

# Instalar dependencias
pip3 install -r requirements.txt

# Configurar cron para apagado automático
echo "*/30 * * * * /opt/clasificador/scripts/check_idle.sh" | crontab -

# Iniciar aplicación
python3 app/main.py 