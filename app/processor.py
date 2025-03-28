import os
import logging
import boto3
import PyPDF2
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
from sqlalchemy import create_engine, text
from config import processing_config, LOGGING_CONFIG
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import spacy
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class NormaProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        
        # Inicializar cliente de Claude
        self.claude = anthropic.Client(api_key=processing_config.anthropic_api_key)
        
        # Inicializar text splitter y embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Claude puede manejar chunks más grandes
            chunk_overlap=400,
            length_function=len
        )
        
        # Usar embeddings de HuggingFace en lugar de OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Template para el prompt de Claude
        self.clasificacion_template = """
        Eres un experto en derecho peruano. Analiza el siguiente texto legal y clasifícalo.
        
        Las categorías disponibles son:
        {categorias}
        
        Texto a analizar:
        {texto}
        
        Proporciona tu respuesta en formato JSON con la siguiente estructura:
        {{
            "categoria_principal": "nombre_categoria",
            "subcategorias": ["subcategoria1", "subcategoria2", "subcategoria3"],
            "palabras_clave": ["palabra1", "palabra2", ..., "palabra10"],
            "resumen": "breve resumen del contenido"
        }}
        
        Asegúrate de que:
        1. La categoría principal sea una de las listadas
        2. Las subcategorías sean específicas y relevantes
        3. Las palabras clave sean términos legales significativos
        4. El resumen capture los puntos principales de la norma
        """
        
        # Definición detallada de categorías y sus criterios
        self.categorias_principales = {
            'derechos_fundamentales': {
                'keywords': ['derecho fundamental', 'derechos humanos', 'garantías constitucionales'],
                'descripcion': """
                Normas relacionadas con:
                - Derechos fundamentales de la persona
                - Libertades constitucionales
                - Garantías constitucionales
                - Derechos humanos
                - Protección de derechos fundamentales
                """
            },
            'administrativo': {
                'keywords': ['procedimiento administrativo', 'gestión pública', 'servicio civil'],
                'descripcion': """
                Normas sobre:
                - Procedimientos administrativos
                - Gestión pública
                - Servicio civil
                - Trámites administrativos
                - Organización del Estado
                """
            },
            'penal': {
                'keywords': ['delito', 'pena', 'sanción penal'],
                'descripcion': """
                Normas relacionadas con:
                - Delitos y penas
                - Proceso penal
                - Sanciones penales
                - Ejecución penal
                - Sistema penitenciario
                """
            },
            'civil': {
                'keywords': ['contrato', 'obligación civil', 'derecho civil'],
                'descripcion': """
                Normas sobre:
                - Contratos civiles
                - Obligaciones
                - Derechos reales
                - Familia
                - Sucesiones
                """
            },
            'laboral': {
                'keywords': ['trabajo', 'empleo', 'relaciones laborales'],
                'descripcion': """
                Normas sobre:
                - Derecho del trabajo
                - Relaciones laborales
                - Seguridad social
                - Pensiones
                - Beneficios laborales
                """
            },
            'tributario': {
                'keywords': ['impuesto', 'tributo', 'contribución'],
                'descripcion': """
                Normas relacionadas con:
                - Impuestos
                - Tributos
                - Contribuciones
                - Procedimientos tributarios
                - Beneficios fiscales
                """
            },
            'ambiental': {
                'keywords': ['medio ambiente', 'recursos naturales', 'conservación'],
                'descripcion': """
                Normas sobre:
                - Protección ambiental
                - Recursos naturales
                - Conservación
                - Cambio climático
                - Gestión ambiental
                """
            }
        }
        
        # Conectar a la base de datos
        self.engine = create_engine(
            f'postgresql://{processing_config.db_user}:{processing_config.db_password}@'
            f'{processing_config.db_host}:{processing_config.db_port}/{processing_config.db_name}'
        )

    def procesar_texto_largo(self, texto: str) -> List[Dict]:
        """
        Procesa textos largos dividiéndolos en chunks y analizándolos con Claude
        """
        # Primero, obtener un resumen general del documento completo
        resumen_general_prompt = """
        Eres un experto en derecho peruano. Lee este documento legal completo y proporciona:
        1. Un resumen ejecutivo
        2. Los puntos principales
        3. El propósito general de la norma
        
        Documento:
        {texto}
        
        Responde en formato JSON:
        {{
            "resumen_ejecutivo": "texto",
            "puntos_principales": ["punto1", "punto2", ...],
            "proposito": "texto"
        }}
        """
        
        try:
            # Obtener resumen general usando los primeros 12000 caracteres
            resumen_response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0,
                messages=[
                    {"role": "user", "content": resumen_general_prompt.format(texto=texto[:12000])}
                ]
            )
            resumen_general = json.loads(resumen_response.content[0].text)
        except Exception as e:
            logger.error(f"Error obteniendo resumen general: {str(e)}")
            resumen_general = None

        # Dividir el texto en chunks manejables
        chunks = self.text_splitter.split_text(texto)
        
        # Crear embeddings y almacenar en FAISS para búsqueda semántica
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        
        # Procesar cada chunk con contexto del resumen general
        resultados = []
        for i, chunk in enumerate(chunks):
            # Obtener categorías como string para el prompt
            categorias_str = "\n".join([
                f"{cat}: {info['descripcion']}"
                for cat, info in self.categorias_principales.items()
            ])
            
            # Incluir contexto del resumen general en el prompt
            contexto_adicional = ""
            if resumen_general:
                contexto_adicional = f"""
                Contexto general del documento:
                Propósito: {resumen_general['proposito']}
                
                Este es el segmento {i+1} de {len(chunks)} del documento completo.
                """
            
            # Preparar el mensaje para Claude con contexto
            mensaje = self.clasificacion_template.format(
                categorias=categorias_str,
                texto=chunk
            ) + contexto_adicional
            
            try:
                # Llamar a Claude
                respuesta = self.claude.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": mensaje}
                    ]
                )
                
                # Extraer y parsear la respuesta JSON
                try:
                    resultado_json = json.loads(respuesta.content[0].text)
                    # Añadir información de contexto
                    if resumen_general and i == 0:  # Solo para el primer chunk
                        resultado_json['resumen_general'] = resumen_general
                    resultado_json['chunk_numero'] = i + 1
                    resultado_json['total_chunks'] = len(chunks)
                    resultados.append(resultado_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decodificando respuesta de Claude: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error en la llamada a Claude: {str(e)}")
                continue
        
        return resultados

    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extraer metadata del documento usando Claude"""
        # Obtener la fecha actual para la extracción
        fecha_actual = datetime.now().date()
        
        # Procesar el texto completo con Claude
        resultados_claude = self.procesar_texto_largo(text)
        
        # Obtener el resumen general si está disponible
        resumen_general = None
        if resultados_claude and resultados_claude[0].get('resumen_general'):
            resumen_general = resultados_claude[0]['resumen_general']
        
        # Consolidar resultados
        categorias_encontradas = {}
        palabras_clave = set()
        resumen_completo = []
        
        for resultado in resultados_claude:
            # Contar categorías
            cat = resultado.get('categoria_principal')
            if cat:
                categorias_encontradas[cat] = categorias_encontradas.get(cat, 0) + 1
            
            # Agregar palabras clave
            palabras_clave.update(resultado.get('palabras_clave', []))
            
            # Agregar al resumen si hay uno
            if resultado.get('resumen'):
                resumen_completo.append(f"Parte {resultado['chunk_numero']}: {resultado['resumen']}")
        
        # Determinar categoría principal (la más frecuente)
        categoria_principal = max(categorias_encontradas.items(), key=lambda x: x[1])[0] if categorias_encontradas else None
        
        metadata = {
            'id': f"NORMA_{os.path.splitext(filename)[0].split('_')[1]}",
            'categoria_principal': categoria_principal,
            'subcategoria_1': resultados_claude[0].get('subcategorias', [None])[0] if resultados_claude else None,
            'subcategoria_2': resultados_claude[0].get('subcategorias', [None, None])[1] if resultados_claude else None,
            'subcategoria_3': resultados_claude[0].get('subcategorias', [None, None, None])[2] if resultados_claude else None,
            'titulo': None,
            'texto_norma': text,
            'palabras_clave': list(palabras_clave)[:10],
            'fuente_url': f"https://diariooficial.elperuano.pe/Normas/obtenerDocumento?idNorma={os.path.splitext(filename)[0].split('_')[1]}",
            'origen': "Diario Oficial El Peruano",
            'nombre_archivo': filename,
            'fecha_extraccion': fecha_actual,
            'estado_vigencia': 'VIGENTE',
            'tipo_norma': None,
            'numero_norma': None,
            'entidad_emisora': None,
            'ambito_aplicacion': 'NACIONAL',
            'referencias_normativas': [],
            'modificaciones': [],
            'observaciones': "\n\n".join([
                "RESUMEN GENERAL:",
                f"Propósito: {resumen_general['proposito'] if resumen_general else 'No disponible'}",
                "\nRESUMEN POR SECCIONES:",
                "\n".join(resumen_completo)
            ])
        }

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

class DocumentAIProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET')
        
        # Inicializar modelos de NLP
        self.nlp = spacy.load("es_core_news_lg")
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Inicializar ChatGPT para análisis avanzado
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Plantilla para análisis de texto
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto en análisis legal y clasificación de documentos. Analiza el siguiente texto y extrae la información relevante."),
            ("user", "Texto del documento: {text}\n\nPor favor, analiza este texto y proporciona:\n1. Resumen ejecutivo\n2. Categoría principal\n3. Palabras clave\n4. Grupo al que pertenece (A, B o C)\n5. Justificación del grupo asignado")
        ])

    def process_document_batch(self, start_date: Optional[str] = None):
        """Procesa un lote de documentos desde una fecha específica"""
        try:
            # Listar documentos PDF en S3
            prefix = 'pdfs/'
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    # Verificar fecha si se especifica
                    if start_date and obj['LastModified'].strftime('%Y-%m-%d') < start_date:
                        continue
                    
                    # Procesar documento
                    doc_key = obj['Key']
                    try:
                        self.process_single_document(doc_key)
                    except Exception as e:
                        logger.error(f"Error procesando documento {doc_key}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes: {str(e)}")

    def process_single_document(self, pdf_key: str):
        """Procesa un documento individual"""
        try:
            # Descargar PDF
            local_path = f"/tmp/{os.path.basename(pdf_key)}"
            self.s3_client.download_file(self.bucket_name, pdf_key, local_path)
            
            # Extraer texto
            text = self.extract_text(local_path)
            if not text:
                logger.warning(f"No se pudo extraer texto de {pdf_key}")
                return
            
            # Analizar con IA
            analysis = self.analyze_text(text)
            
            # Actualizar JSON correspondiente
            self.update_document_metadata(pdf_key, analysis)
            
            # Limpiar
            os.remove(local_path)
            
        except Exception as e:
            logger.error(f"Error procesando documento {pdf_key}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)

    def extract_text(self, pdf_path: str) -> str:
        """Extrae texto de un PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF: {str(e)}")
            return ""

    def analyze_text(self, text: str) -> Dict:
        """Analiza el texto usando múltiples modelos de IA"""
        try:
            # Análisis con ChatGPT
            chain = self.analysis_prompt | self.chat_model
            gpt_analysis = chain.invoke({"text": text[:4000]})  # Limitar longitud
            
            # Análisis con spaCy para entidades y frases clave
            doc = self.nlp(text[:10000])  # Limitar para rendimiento
            entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'LAW', 'DATE']]
            
            # Clasificación con zero-shot
            categories = ['CONSTITUCIONAL', 'ADMINISTRATIVO', 'PENAL', 'CIVIL', 'LABORAL']
            classification = self.zero_shot(text[:1000], categories, multi_label=True)
            
            return {
                'gpt_analysis': gpt_analysis.content,
                'entities': entities,
                'classification': {
                    'labels': classification['labels'],
                    'scores': classification['scores']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de texto: {str(e)}")
            return {}

    def update_document_metadata(self, pdf_key: str, analysis: Dict):
        """Actualiza el JSON de metadata con el análisis de IA"""
        try:
            # Construir key del JSON
            json_key = pdf_key.replace('pdfs/', 'metadata/').replace('.pdf', '.json')
            
            try:
                # Intentar obtener JSON existente
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=json_key)
                metadata = json.loads(response['Body'].read().decode('utf-8'))
            except:
                metadata = {}
            
            # Actualizar con análisis de IA
            metadata.update({
                'ai_analysis': analysis,
                'last_updated': datetime.now().isoformat()
            })
            
            # Guardar JSON actualizado
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=json_key,
                Body=json.dumps(metadata, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Metadata actualizada para {pdf_key}")
            
        except Exception as e:
            logger.error(f"Error actualizando metadata: {str(e)}")

def main():
    """Función principal"""
    processor = NormaProcessor()
    # Aquí agregaremos la lógica para procesar documentos en batch
    
if __name__ == "__main__":
    main() 