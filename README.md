# Web Scraping del Diario Oficial El Peruano

Este proyecto realiza web scraping del Diario Oficial El Peruano para extraer PDFs de normas legales y procesar su contenido para entrenamiento de IA.

## Características

- Extracción automática de PDFs del Diario Oficial
- Procesamiento de texto de los PDFs
- Generación de resúmenes usando IA
- Clasificación de contenido por tipo de norma
- Almacenamiento de datos en formatos JSON y CSV
- Sistema de logging para seguimiento de errores

## Requisitos

- Python 3.8 o superior
- Chrome browser instalado
- Conexión a internet
- Espacio suficiente en disco para almacenar PDFs

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_DIRECTORIO]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar el script principal:
```bash
python scraper.py
```

2. Los resultados se guardarán en:
   - `pdfs/`: Directorio con los PDFs descargados
   - `data/`: Directorio con los datos procesados
     - `normas.json`: Datos completos en formato JSON
     - `normas.csv`: Datos en formato CSV para análisis
   - `scraping.log`: Registro de la ejecución

## Estructura de Datos

Los datos extraídos incluyen:
- URL del PDF
- Título del documento
- Fecha de extracción
- Contenido completo
- Resumen generado por IA
- Categorías clasificadas por IA

## Notas

- El script utiliza Selenium para la automatización del navegador
- Se implementa un sistema de espera para asegurar la carga completa de la página
- Los PDFs se procesan secuencialmente para evitar sobrecarga
- Se incluye manejo de errores y logging para seguimiento

## Limitaciones

- La velocidad de extracción depende de la velocidad de internet
- El procesamiento de PDFs puede tomar tiempo según su tamaño
- La clasificación de IA puede no ser 100% precisa 