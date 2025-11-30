import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify, render_template
import requests
import time

# --- CONFIGURACIÓN DEL MODELO ---
MODEL_FILE = "anime_similarity_matrix.pkl"
item_similarity_df = None

# --- LÓGICA DE CARGA DEL MODELO ---

def load_model():
    """
    Carga la matriz de similitud de ítems (nuestro "modelo") al iniciar la aplicación.
    """
    global item_similarity_df
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as file:
                item_similarity_df = pickle.load(file)
            print(f"✅ Matriz de Similitud cargada. Tamaño: {item_similarity_df.shape}")
            return True
        except Exception as e:
            print(f"❌ ERROR: No se pudo cargar el archivo de modelo '{MODEL_FILE}': {e}")
            return False
    else:
        print(f"❌ ERROR: Archivo de modelo '{MODEL_FILE}' no encontrado.")
        print("Asegúrate de ejecutar el script de entrenamiento primero para generar la matriz.")
        return False

# --- FUNCIÓN DE RECOMENDACIÓN ---

def recommend_animes(anime_name, similarity_df, top_n=10):
    """
    Recomienda animes basados en la similitud de coseno con un anime dado.
    """
    if anime_name not in similarity_df.index:
        # Devolvemos un DataFrame vacío y un código de error si no se encuentra
        return pd.Series(), 404

    # Obtener las puntuaciones de similitud para el anime_name
    similar_scores = similarity_df[anime_name]

    # Ordenar los animes por puntuación de similitud de forma descendente
    # Excluir el propio anime_name (similitud 1.0)
    recommendations = similar_scores.sort_values(ascending=False)
    recommendations = recommendations.drop(labels=anime_name, errors='ignore')

    # Devolver los N principales
    return recommendations.head(top_n), 200

# --- CONFIGURACIÓN Y ENDPOINT DE FLASK ---
def get_anime_image_url(anime_name):
    """
    Busca el anime en Jikan (MyAnimeList) y devuelve la URL del poster.
    Si falla, espera un momento y vuelve a intentar (manejo de Rate Limit).
    """
    url = f"https://api.jikan.moe/v4/anime?q={anime_name}"
    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES):
        try:
            # 1. Realizar la solicitud HTTP
            response = requests.get(url, timeout=5)
            response.raise_for_status() # Lanza error para 4xx/5xx

            data = response.json()
            
            # 2. Manejar el límite de tasa 429 (Too Many Requests)
            if response.status_code == 429:
                print(f"⚠️ Rate Limit alcanzado para '{anime_name}'. Esperando 5 segundos...")
                time.sleep(5)
                continue # Reintentar la solicitud

            # 3. Extraer la URL de la imagen si hay datos
            if "data" in data and len(data["data"]) > 0:
                # Usamos la imagen JPG de tamaño medio/por defecto
                return data["data"][0]["images"]["jpg"]["image_url"]

            # Si no hay datos, salimos del bucle
            return None 

        except requests.exceptions.HTTPError as e:
            print(f"Error HTTP ({response.status_code}) para '{anime_name}': {e}")
            if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                print("Intentando nuevamente...")
                time.sleep(5)
                continue
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión para '{anime_name}': {e}")
            return None
            
    # Devuelve la URL de placeholder si falla después de todos los reintentos
    return "https://via.placeholder.com/150x200?text=No+Image"


app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación Flask
if not load_model():
    # Si la carga falla, no tiene sentido iniciar la API
    print("API no iniciada debido a que el modelo no pudo cargarse.")
    exit()

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/recommend', methods=['GET']) 
def get_recommendations():
    """
    Endpoint para obtener recomendaciones y enriquecer con la URL de la imagen (Jikan).
    """
    anime_name = request.args.get('anime')
    
    if not anime_name:
        return jsonify({"error": "Parámetro faltante", "message": "Debes especificar el nombre de un anime usando '?anime=<nombre>'"}), 400

    # Obtener las recomendaciones del modelo
    recommendations_series, status_code = recommend_animes(
        anime_name, 
        item_similarity_df, 
        top_n=10
    )
    
    if status_code == 404:
        return jsonify({"error": "Anime no encontrado", "message": f"El anime '{anime_name}' no está en la base de datos de animes populares."}), 404
        
    # Formatear la respuesta, AÑADIENDO la URL de la imagen
    recommendations_list = []
    for name, score in recommendations_series.items():
        image_url = get_anime_image_url(name) 
        
        recommendations_list.append({
            "name": name, 
            "similarity_score": round(score, 4),
            "image_url": image_url if image_url else "https://via.placeholder.com/150x200?text=No+Image" # Usar placeholder si es None
        })

    return jsonify({"input_anime": anime_name, "recommendations": recommendations_list}), 200

# --- INICIO DEL SERVIDOR ---

if __name__ == '__main__':
    # Usamos host='0.0.0.0' para hacerlo accesible externamente si es necesario
    # debug=True es útil durante el desarrollo
    app.run(debug=True, host='0.0.0.0', port=5000)