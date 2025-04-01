import re
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextStructureProcessor:
    def __init__(self):
        """Inicializa el procesador de estructura de texto"""
        # Patrones para identificar secciones
        self.patrones = {
            'titulo': r'TÍTULO\s+([IVX]+)[:\s-]+(.+?)(?=TÍTULO|\Z)',
            'capitulo': r'CAPÍTULO\s+([IVX]+)[:\s-]+(.+?)(?=CAPÍTULO|\Z)',
            'seccion': r'SECCIÓN\s+([IVX]+)[:\s-]+(.+?)(?=SECCIÓN|\Z)',
            'articulo': r'Artículo\s+(\d+)[°]?\.?[-:]?\s*(.+?)(?=Artículo|\Z)',
            'tipo_norma': r'(LEY|DECRETO SUPREMO|RESOLUCIÓN|ORDENANZA)\s+N°\s*(\d+[-\d\w]*)',
        }
        
        # Categorías y palabras clave
        self.categorias = {
            'CONSTITUCIONAL': ['constitución', 'constitucional', 'derechos fundamentales'],
            'ADMINISTRATIVO': ['administrativo', 'administración pública', 'procedimiento'],
            'PENAL': ['penal', 'delito', 'sanción', 'pena'],
            'CIVIL': ['civil', 'contratos', 'obligaciones'],
            'LABORAL': ['trabajo', 'laboral', 'trabajador'],
            'TRIBUTARIO': ['tributo', 'impuesto', 'contribución'],
            'AMBIENTAL': ['ambiental', 'ambiente', 'ecológico']
        }

    def extract_structure(self, text: str) -> dict:
        """Extrae la estructura jerárquica del texto"""
        estructura = {
            'titulos': [],
            'capitulos': [],
            'secciones': [],
            'articulos': [],
            'metadata': {}
        }
        
        # Extraer títulos
        for match in re.finditer(self.patrones['titulo'], text, re.DOTALL):
            estructura['titulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            })
            
        # Extraer capítulos
        for match in re.finditer(self.patrones['capitulo'], text, re.DOTALL):
            estructura['capitulos'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            })
            
        # Extraer secciones
        for match in re.finditer(self.patrones['seccion'], text, re.DOTALL):
            estructura['secciones'].append({
                'numero': match.group(1),
                'nombre': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            })
            
        # Extraer artículos
        for match in re.finditer(self.patrones['articulo'], text, re.DOTALL):
            estructura['articulos'].append({
                'numero': match.group(1),
                'contenido': match.group(2).strip(),
                'texto': match.group(0),
                'inicio': match.start(),
                'fin': match.end()
            })
            
        return estructura

    def determine_category(self, text: str) -> tuple:
        """Determina la categoría principal y palabras clave del texto"""
        text = text.lower()
        scores = {cat: 0 for cat in self.categorias}
        found_keywords = set()
        
        # Contar ocurrencias de palabras clave
        for categoria, keywords in self.categorias.items():
            for keyword in keywords:
                if keyword in text:
                    scores[categoria] += 1
                    found_keywords.add(keyword)
        
        # Determinar categoría principal
        if not any(scores.values()):
            return 'OTROS', list(found_keywords)
        
        categoria_principal = max(scores.items(), key=lambda x: x[1])[0]
        return categoria_principal, list(found_keywords)

    def extract_metadata(self, text: str) -> dict:
        """Extrae metadatos del texto"""
        metadata = {
            'tipo_norma': None,
            'numero_norma': None,
            'fecha_extraccion': datetime.now().date().isoformat(),
            'estado_vigencia': 'VIGENTE',  # Por defecto
            'entidad_emisora': None,
            'ambito_aplicacion': 'NACIONAL'  # Por defecto
        }
        
        # Extraer tipo y número de norma
        match = re.search(self.patrones['tipo_norma'], text)
        if match:
            metadata['tipo_norma'] = match.group(1)
            metadata['numero_norma'] = match.group(2)
        
        return metadata

    def process_document(self, text: str) -> dict:
        """Procesa el documento completo y retorna la estructura"""
        # Extraer estructura jerárquica
        estructura = self.extract_structure(text)
        
        # Determinar categoría y palabras clave
        categoria, keywords = self.determine_category(text)
        
        # Extraer metadatos
        metadata = self.extract_metadata(text)
        
        # Crear documento estructurado
        documento = {
            'id': f"NORMA_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'categoria_principal': categoria,
            'subcategoria_1': None,
            'subcategoria_2': None,
            'subcategoria_3': None,
            'titulo_numero': estructura['titulos'][0]['numero'] if estructura['titulos'] else None,
            'titulo_nombre': estructura['titulos'][0]['nombre'] if estructura['titulos'] else None,
            'capitulo_numero': estructura['capitulos'][0]['numero'] if estructura['capitulos'] else None,
            'capitulo_nombre': estructura['capitulos'][0]['nombre'] if estructura['capitulos'] else None,
            'seccion_numero': estructura['secciones'][0]['numero'] if estructura['secciones'] else None,
            'seccion_nombre': estructura['secciones'][0]['nombre'] if estructura['secciones'] else None,
            'articulo': estructura['articulos'][0]['numero'] if estructura['articulos'] else None,
            'titulo': estructura['titulos'][0]['nombre'] if estructura['titulos'] else None,
            'palabras_clave': keywords,
            'texto_norma': text,
            **metadata  # Incluir todos los metadatos extraídos
        }
        
        return documento 