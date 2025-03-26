import os
import logging
import boto3
import PyPDF2
import json
from datetime import datetime
from typing import Dict, Any, List
import re
from sqlalchemy import create_engine, text
from config import processing_config, LOGGING_CONFIG

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class NormaProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        # Categorías principales y sus palabras clave
        self.categorias_principales = {
            'derechos_fundamentales': ['derecho fundamental', 'derechos humanos', 'garantías constitucionales'],
            'administrativo': ['procedimiento administrativo', 'gestión pública', 'servicio civil'],
            'penal': ['delito', 'pena', 'sanción penal'],
            'civil': ['contrato', 'obligación civil', 'derecho civil'],
            'laboral': ['trabajo', 'empleo', 'relaciones laborales'],
            'tributario': ['impuesto', 'tributo', 'contribución'],
            'ambiental': ['medio ambiente', 'recursos naturales', 'conservación']
        }
        
        # Conectar a la base de datos
        self.engine = create_engine(
            f'postgresql://{processing_config.db_user}:{processing_config.db_password}@'
            f'{processing_config.db_host}:{processing_config.db_port}/{processing_config.db_name}'
        )

    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extraer metadata del documento"""
        # Obtener la fecha actual para la extracción
        fecha_actual = datetime.now().date()
        
        # Patrones para extraer información
        patrones = {
            'tipo_norma': r'(RESOLUCIÓN|DECRETO|LEY|ORDENANZA)\s+(SUPREMA|LEGISLATIVA|MUNICIPAL)?',
            'numero_norma': r'N°\s*(\d+[-/\w]*)',
            'entidad_emisora': r'(MINISTERIO|CONGRESO|MUNICIPALIDAD|GOBIERNO REGIONAL)\s+DE\s+[\w\s]+',
        }
        
        metadata = {
            'id': f"NORMA_{os.path.splitext(filename)[0].split('_')[1]}",
            'categoria_principal': None,
            'subcategoria_1': None,
            'subcategoria_2': None,
            'subcategoria_3': None,
            'articulo': None,
            'titulo': None,
            'texto': text,
            'palabras_clave': [],
            'fuente': f"https://diariooficial.elperuano.pe/Normas/obtenerDocumento?idNorma={os.path.splitext(filename)[0].split('_')[1]}",
            'origen': "Diario Oficial El Peruano",
            'nombre_archivo': filename,
            'fecha_extraccion': fecha_actual,
            'estado_vigencia': 'VIGENTE',  # Por defecto asumimos que está vigente
            'tipo_norma': None,
            'numero_norma': None,
            'entidad_emisora': None,
            'ambito_aplicacion': 'NACIONAL',  # Por defecto
            'referencias_normativas': [],
            'modificaciones': [],
            'observaciones': None
        }

        # Detectar categoría principal
        for categoria, keywords in self.categorias_principales.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    metadata['categoria_principal'] = categoria
                    break
            if metadata['categoria_principal']:
                break

        # Extraer título (primera línea no vacía)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            metadata['titulo'] = lines[0]

        # Extraer tipo de norma y número
        for key, patron in patrones.items():
            match = re.search(patron, text, re.IGNORECASE)
            if match:
                if key == 'tipo_norma':
                    metadata[key] = ' '.join(filter(None, match.groups())).upper()
                elif key == 'numero_norma':
                    metadata[key] = match.group(1)
                elif key == 'entidad_emisora':
                    metadata[key] = match.group(0).strip().upper()

        # Buscar referencias a otras normas
        referencias = re.findall(r'(LEY|DECRETO|RESOLUCIÓN)\s+N°\s*\d+[-/\w]*', text, re.IGNORECASE)
        metadata['referencias_normativas'] = [ref.strip() for ref in referencias]

        # Extraer palabras clave significativas
        palabras_clave = set()
        for palabra in text.lower().split():
            if len(palabra) > 4:  # Ignorar palabras muy cortas
                palabras_clave.add(palabra)
        metadata['palabras_clave'] = list(palabras_clave)[:10]  # Limitar a 10 palabras clave

        return metadata

    def save_to_database(self, metadata: Dict[str, Any]):
        """Guardar metadata en la base de datos"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO normas_legales (
                        id, categoria_principal, subcategoria_1, subcategoria_2, 
                        subcategoria_3, articulo, titulo, texto_norma,
                        palabras_clave, fuente_url, origen, nombre_archivo,
                        fecha_procesamiento, fecha_extraccion, estado_vigencia,
                        tipo_norma, numero_norma, entidad_emisora, ambito_aplicacion,
                        referencias_normativas, modificaciones, observaciones
                    ) VALUES (
                        :id, :categoria_principal, :subcategoria_1, :subcategoria_2,
                        :subcategoria_3, :articulo, :titulo, :texto,
                        :palabras_clave, :fuente, :origen, :nombre_archivo,
                        :fecha_procesamiento, :fecha_extraccion, :estado_vigencia,
                        :tipo_norma, :numero_norma, :entidad_emisora, :ambito_aplicacion,
                        :referencias_normativas, :modificaciones, :observaciones
                    )
                """)
                
                conn.execute(
                    query,
                    {
                        **metadata,
                        'fecha_procesamiento': datetime.now(),
                        'palabras_clave': json.dumps(metadata['palabras_clave']),
                        'referencias_normativas': json.dumps(metadata['referencias_normativas']),
                        'modificaciones': json.dumps(metadata['modificaciones'])
                    }
                )
                conn.commit()
                logger.info(f"Norma {metadata['id']} guardada exitosamente")
                
        except Exception as e:
            logger.error(f"Error guardando norma en la base de datos: {str(e)}")
            raise

    def process_document(self, bucket: str, key: str):
        """Procesar un documento PDF"""
        try:
            # Descargar PDF de S3
            local_path = f"/tmp/{os.path.basename(key)}"
            self.s3_client.download_file(bucket, key, local_path)
            
            # Extraer texto
            text = ""
            with open(local_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            
            # Extraer y guardar metadata
            metadata = self.extract_metadata(text, os.path.basename(key))
            self.save_to_database(metadata)
            
            # Limpiar
            os.remove(local_path)
            
            logger.info(f"Documento procesado exitosamente: {key}")
            
        except Exception as e:
            logger.error(f"Error procesando documento {key}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)

def main():
    """Función principal"""
    processor = NormaProcessor()
    # Aquí agregaremos la lógica para procesar documentos en batch
    
if __name__ == "__main__":
    main() 