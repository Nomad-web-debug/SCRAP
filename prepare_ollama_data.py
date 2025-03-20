import json
import pandas as pd
from typing import List, Dict
import os

def prepare_training_data(csv_path: str, output_path: str):
    """Preparar datos para entrenamiento con Ollama"""
    # Leer el CSV
    df = pd.read_csv(csv_path)
    
    # Crear formato de entrenamiento
    training_data = []
    
    for _, row in df.iterrows():
        # Crear prompt
        prompt = f"""Analiza la siguiente norma legal peruana:

Título: {row['Título']}
Tipo: {row['Tipo de Norma']}
Fecha: {row['Fecha de Publicación']}
Referencia: {row['Referencia']}

Organizaciones mencionadas: {', '.join(row['Organizaciones'])}
Personas mencionadas: {', '.join(row['Personas'])}
Lugares mencionados: {', '.join(row['Lugares'])}

Objetivos:
{chr(10).join(row['Objetivos'])}

Artículos:
{chr(10).join([f"Artículo {art['numero']}: {art['contenido']}" for art in row['Artículos']])}

Resumen: {row['Resumen']}

Palabras clave: {', '.join(row['Palabras Clave'])}"""

        # Crear respuesta esperada
        response = f"""Esta norma legal es un {row['Tipo de Norma']} que fue publicada el {row['Fecha de Publicación']}.

Los principales objetivos son:
{chr(10).join(row['Objetivos'])}

La norma afecta a las siguientes organizaciones:
{chr(10).join([f"- {org}" for org in row['Organizaciones']])}

Los puntos clave son:
{chr(10).join([f"- {art['contenido']}" for art in row['Artículos'][:3]])}

Esta norma está relacionada con los siguientes temas:
{chr(10).join([f"- {kw} ({score:.2f})" for kw, score in zip(row['Palabras Clave'], row['Puntuaciones de Keywords'])])}"""

        # Agregar al conjunto de entrenamiento
        training_data.append({
            "prompt": prompt,
            "response": response
        })
    
    # Guardar datos de entrenamiento
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

def create_ollama_model(model_name: str = "peruano-legal"):
    """Crear modelo de Ollama"""
    # Crear archivo de configuración del modelo
    model_config = {
        "name": model_name,
        "model": "llama2",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,
            "num_thread": 4
        }
    }
    
    # Guardar configuración
    with open(f"{model_name}.json", 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)
    
    # Crear comando para entrenar el modelo
    print(f"\nPara entrenar el modelo, ejecuta:")
    print(f"ollama create {model_name} -f {model_name}.json")
    print(f"ollama train {model_name} training_data.json")

if __name__ == "__main__":
    # Configurar rutas
    csv_path = "data/normas_metadata.csv"
    output_path = "data/training_data.json"
    
    # Crear directorio si no existe
    os.makedirs("data", exist_ok=True)
    
    # Preparar datos
    print("Preparando datos para entrenamiento...")
    prepare_training_data(csv_path, output_path)
    
    # Crear modelo
    print("\nCreando configuración del modelo...")
    create_ollama_model()
    
    print("\n¡Proceso completado!")
    print(f"Los datos de entrenamiento se han guardado en: {output_path}") 