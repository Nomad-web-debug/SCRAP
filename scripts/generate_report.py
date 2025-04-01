import json
import os
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_files(output_dir: str) -> List[Dict]:
    """
    Carga todos los archivos JSON procesados
    """
    results = []
    
    try:
        # Buscar archivos JSON
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            try:
                with open(os.path.join(output_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logger.error(f"Error leyendo {json_file}: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Error cargando archivos procesados: {str(e)}")
        return []

def generate_summary(results: List[Dict]) -> Dict:
    """
    Genera un resumen del procesamiento
    """
    summary = {
        'total_documentos': len(results),
        'documentos_por_tipo': {},
        'documentos_por_rama': {},
        'promedio_palabras_clave': 0,
        'documentos_con_estructura': 0,
        'documentos_modificados': 0
    }
    
    total_tags = 0
    
    for doc in results:
        # Contar por tipo
        tipo = doc.get('tipo_documento', 'DESCONOCIDO')
        summary['documentos_por_tipo'][tipo] = summary['documentos_por_tipo'].get(tipo, 0) + 1
        
        # Contar por rama
        rama = doc.get('rama_derecho', 'DESCONOCIDA')
        summary['documentos_por_rama'][rama] = summary['documentos_por_rama'].get(rama, 0) + 1
        
        # Contar tags
        tags = doc.get('tags', [])
        total_tags += len(tags)
        
        # Verificar estructura
        if doc.get('titulo') or doc.get('capitulo'):
            summary['documentos_con_estructura'] += 1
            
        # Verificar modificaciones
        if doc.get('modificado', False):
            summary['documentos_modificados'] += 1
    
    # Calcular promedio de palabras clave
    if summary['total_documentos'] > 0:
        summary['promedio_palabras_clave'] = total_tags / summary['total_documentos']
    
    return summary

def generate_excel_report(results: List[Dict], summary: Dict, output_dir: str):
    """
    Genera un reporte Excel con los resultados
    """
    try:
        # Crear DataFrame con resultados
        df_docs = pd.DataFrame(results)
        
        # Crear DataFrame con resumen
        df_summary = pd.DataFrame([{
            'Total Documentos': summary['total_documentos'],
            'Documentos con Estructura': summary['documentos_con_estructura'],
            'Documentos Modificados': summary['documentos_modificados'],
            'Promedio Palabras Clave': f"{summary['promedio_palabras_clave']:.2f}"
        }])
        
        # Crear DataFrames para distribuciones
        df_tipos = pd.DataFrame(list(summary['documentos_por_tipo'].items()), 
                              columns=['Tipo', 'Cantidad'])
        df_ramas = pd.DataFrame(list(summary['documentos_por_rama'].items()),
                              columns=['Rama', 'Cantidad'])
        
        # Crear archivo Excel
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = os.path.join(output_dir, f'reporte_procesamiento_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Escribir hojas
            df_summary.to_excel(writer, sheet_name='Resumen', index=False)
            df_tipos.to_excel(writer, sheet_name='Por Tipo', index=False)
            df_ramas.to_excel(writer, sheet_name='Por Rama', index=False)
            df_docs.to_excel(writer, sheet_name='Documentos', index=False)
            
            # Dar formato
            workbook = writer.book
            
            # Formato para números
            num_format = workbook.add_format({'num_format': '#,##0'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Ajustar columnas
            for sheet in writer.sheets.values():
                sheet.autofit()
        
        logger.info(f"Reporte Excel generado: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error generando reporte Excel: {str(e)}")
        return None

def generate_text_report(summary: Dict, output_dir: str):
    """
    Genera un reporte en texto plano
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'reporte_procesamiento_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE PROCESAMIENTO DE DOCUMENTOS LEGALES\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("RESUMEN GENERAL\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total de documentos procesados: {summary['total_documentos']}\n")
            f.write(f"Documentos con estructura: {summary['documentos_con_estructura']}\n")
            f.write(f"Documentos modificados: {summary['documentos_modificados']}\n")
            f.write(f"Promedio de palabras clave: {summary['promedio_palabras_clave']:.2f}\n\n")
            
            f.write("DISTRIBUCIÓN POR TIPO DE DOCUMENTO\n")
            f.write("-" * 20 + "\n")
            for tipo, cantidad in summary['documentos_por_tipo'].items():
                f.write(f"{tipo}: {cantidad}\n")
            f.write("\n")
            
            f.write("DISTRIBUCIÓN POR RAMA DEL DERECHO\n")
            f.write("-" * 20 + "\n")
            for rama, cantidad in summary['documentos_por_rama'].items():
                f.write(f"{rama}: {cantidad}\n")
            
        logger.info(f"Reporte de texto generado: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generando reporte de texto: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generador de reportes de procesamiento')
    parser.add_argument('--input-dir', required=True, help='Directorio con PDFs originales')
    parser.add_argument('--output-dir', required=True, help='Directorio con resultados procesados')
    args = parser.parse_args()
    
    try:
        # Cargar resultados
        results = load_processed_files(args.output_dir)
        if not results:
            logger.error("No se encontraron resultados procesados")
            return
            
        # Generar resumen
        summary = generate_summary(results)
        
        # Generar reportes
        excel_path = generate_excel_report(results, summary, args.output_dir)
        text_path = generate_text_report(summary, args.output_dir)
        
        if excel_path and text_path:
            logger.info("Reportes generados exitosamente")
            logger.info(f"Excel: {excel_path}")
            logger.info(f"Texto: {text_path}")
        else:
            logger.error("Error generando algunos reportes")
            
    except Exception as e:
        logger.error(f"Error en el proceso de generación de reportes: {str(e)}")

if __name__ == "__main__":
    main() 