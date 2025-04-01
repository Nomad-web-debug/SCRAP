import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextStructureProcessor:
    def __init__(self):
        """Inicializa el procesador de estructura de texto"""
        # Patrones para identificar secciones
        self.patrones = {
            'titulo': r'T[IÍ]TULO\s+([IVX]+)[:\s-]+(.+?)(?=T[IÍ]TULO|\Z)',
            'capitulo': r'CAP[IÍ]TULO\s+([IVX]+)[:\s-]+(.+?)(?=CAP[IÍ]TULO|\Z)',
            'seccion': r'SECCI[OÓ]N\s+([IVX]+)[:\s-]+(.+?)(?=SECCI[OÓ]N|\Z)',
            'articulo': r'Art[ií]culo\s+(\d+)[°]?\.?[-:]?\s*(.+?)(?=Art[ií]culo|\Z)',
            'tipo_norma': r'(LEY|DECRETO SUPREMO|RESOLUCIÓN|ORDENANZA)\s+N°\s*(\d+[-\d\w]*)',
        }
        
        # Categorías principales y sus palabras clave
        self.categorias = {
            'CONSTITUCIONAL': [
                'constitución', 'constitucional', 'derechos fundamentales',
                'garantías constitucionales', 'reforma constitucional'
            ],
            'ADMINISTRATIVO': [
                'administrativo', 'administración pública', 'procedimiento',
                'servicio público', 'función pública', 'gestión pública'
            ],
            'PENAL': [
                'penal', 'delito', 'sanción', 'pena', 'procesal penal',
                'criminal', 'delincuencia', 'prisión'
            ],
            'CIVIL': [
                'civil', 'contratos', 'obligaciones', 'personas', 'familia',
                'propiedad', 'sucesión', 'responsabilidad civil'
            ],
            'LABORAL': [
                'trabajo', 'laboral', 'trabajador', 'empleo', 'sindical',
                'seguridad social', 'pensiones', 'beneficios laborales'
            ],
            'TRIBUTARIO': [
                'tributo', 'impuesto', 'contribución', 'fiscal', 'tributación',
                'recaudación', 'sunat', 'obligación tributaria'
            ],
            'AMBIENTAL': [
                'ambiental', 'ambiente', 'ecológico', 'recursos naturales',
                'conservación', 'biodiversidad', 'contaminación'
            ]
        }

    def extract_structure(self, text: str) -> Dict:
        """
        Extrae la estructura jerárquica del texto legal
        """
        estructura = {
            'titulos': [],
            'capitulos': [],
            'secciones': [],
            'articulos': [],
            'metadata': {}
        }
        
        # Procesar el texto para cada patrón
        for tipo, patron in self.patrones.items():
            if tipo == 'tipo_norma':
                continue
                
            for match in re.finditer(patron, text, re.DOTALL | re.IGNORECASE):
                item = {
                    'numero': match.group(1),
                    'nombre': match.group(2).strip() if len(match.groups()) > 1 else '',
                    'texto': match.group(0),
                    'inicio': match.start(),
                    'fin': match.end()
                }
                
                if tipo == 'articulo':
                    item['contenido'] = item['nombre']
                    
                estructura[f"{tipo}s"].append(item)
        
        # Ordenar elementos por posición
        for key in estructura.keys():
            if isinstance(estructura[key], list):
                estructura[key].sort(key=lambda x: x.get('inicio', 0))
        
        return estructura

    def determine_category(self, text: str) -> Tuple[str, List[str], Dict[str, float]]:
        """
        Determina la categoría principal, subcategorías y palabras clave
        """
        text = text.lower()
        scores = {cat: 0 for cat in self.categorias}
        found_keywords = set()
        
        # Analizar texto para cada categoría
        for categoria, keywords in self.categorias.items():
            for keyword in keywords:
                if keyword in text:
                    scores[categoria] += 1
                    found_keywords.add(keyword)
        
        # Calcular porcentajes de confianza
        total_matches = sum(scores.values())
        if total_matches > 0:
            confidence = {cat: (score / total_matches) * 100 
                        for cat, score in scores.items()}
        else:
            confidence = {cat: 0 for cat in scores}
        
        # Determinar categoría principal y subcategorías
        sorted_categories = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        main_category = sorted_categories[0][0] if sorted_categories else 'OTROS'
        
        # Obtener subcategorías (las siguientes 3 categorías más relevantes)
        subcategories = [cat for cat, _ in sorted_categories[1:4]]
        while len(subcategories) < 3:
            subcategories.append(None)
        
        return main_category, list(found_keywords), dict(zip(['subcategoria_1', 'subcategoria_2', 'subcategoria_3'], subcategories))

    def extract_metadata(self, text: str) -> Dict:
        """
        Extrae metadatos del texto legal
        """
        metadata = {
            'tipo_norma': None,
            'numero_norma': None,
            'fecha_extraccion': datetime.now().date().isoformat(),
            'estado_vigencia': 'VIGENTE',
            'entidad_emisora': None,
            'ambito_aplicacion': 'NACIONAL',
            'referencias_normativas': [],
            'modificaciones': []
        }
        
        # Extraer tipo y número de norma
        match = re.search(self.patrones['tipo_norma'], text, re.IGNORECASE)
        if match:
            metadata['tipo_norma'] = match.group(1).upper()
            metadata['numero_norma'] = match.group(2)
            
        # Buscar referencias a otras normas
        referencias_pattern = r'(LEY|DECRETO SUPREMO|RESOLUCIÓN|ORDENANZA)\s+N°\s*(\d+[-\d\w]*)'
        for match in re.finditer(referencias_pattern, text):
            ref = f"{match.group(1)} N° {match.group(2)}"
            if ref not in metadata['referencias_normativas']:
                metadata['referencias_normativas'].append(ref)
        
        return metadata

    def process_document(self, text: str) -> Dict:
        """
        Procesa el documento completo y retorna la estructura
        """
        # Extraer estructura jerárquica
        estructura = self.extract_structure(text)
        
        # Determinar categoría y palabras clave
        categoria, keywords, subcategorias = self.determine_category(text)
        
        # Extraer metadatos
        metadata = self.extract_metadata(text)
        
        # Crear documento estructurado
        documento = {
            'id': f"NORMA_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'categoria_principal': categoria,
            **subcategorias,  # Incluye subcategoria_1, subcategoria_2, subcategoria_3
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