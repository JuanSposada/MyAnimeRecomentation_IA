import pandas as pd
import pickle
import os
import sys # Útil para salir del script si no hay matriz disponible

# --- CONFIGURACIÓN ---
MODEL_FILE = "anime_similarity_matrix.pkl"
item_similarity_df = None

# --- LÓGICA DE CARGA Y ENTRENAMIENTO CONDICIONAL ---

if os.path.exists(MODEL_FILE):
    # 1. INTENTAR CARGAR EL MODELO
    print(f"✅ Archivo de modelo encontrado: '{MODEL_FILE}'. Cargando...")
    try:
        with open(MODEL_FILE, 'rb') as file:
            item_similarity_df = pickle.load(file)
        print(f"Matriz de Similitud cargada. Tamaño: {item_similarity_df.shape}")
        
    except Exception as e:
        # Si hay un error (por ejemplo, el archivo está corrupto), lo notifica y fuerza el re-entrenamiento
        print(f"❌ Error al cargar el archivo de modelo: {e}. Se procederá a generar uno nuevo.")
        item_similarity_df = None
        
else:
    print(f"❌ Archivo de modelo no encontrado: '{MODEL_FILE}'. Se procederá a generar la matriz (ENTRENAMIENTO).")


if item_similarity_df is None:
    # 2. BLOQUE DE ENTRENAMIENTO (SOLO si el modelo no pudo cargarse)
    
    # Importaciones que solo son necesarias para el entrenamiento
    try:
        import kagglehub
        from scipy.sparse import csr_matrix
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("\n¡ERROR! Faltan librerías necesarias para el entrenamiento (kagglehub, scipy, sklearn).")
        sys.exit(1) # Detiene la ejecución si no puede entrenar
        
    print("\n--- INICIANDO PROCESO DE ENTRENAMIENTO ---")
    
    # Descarga y Carga de datos
    # La ruta donde se descargó el dataset
    dataset_path = kagglehub.dataset_download("CooperUnion/anime-recommendations-database")
    ratings_file = f"{dataset_path}/rating.csv"
    anime_file = f"{dataset_path}/anime.csv"

    # Cargar los DataFrames
    df_ratings = pd.read_csv(ratings_file)
    df_anime = pd.read_csv(anime_file)
    print("DataFrames de entrada cargados.")

    # Limpieza y Filtro
    df_ratings_clean = df_ratings[df_ratings['rating'] != -1]
    df_merged = pd.merge(df_ratings_clean, df_anime[['anime_id', 'name']], on='anime_id', how='inner')

    # Filtro por popularidad (min 50 calificaciones)
    anime_rating_counts = df_merged.groupby('name')['rating'].count()
    popular_animes = anime_rating_counts[anime_rating_counts >= 50].index
    df_filtered_anime = df_merged[df_merged['name'].isin(popular_animes)]

    # Filtro por usuarios activos (min 50 calificaciones)
    user_rating_counts = df_filtered_anime.groupby('user_id')['rating'].count()
    active_users = user_rating_counts[user_rating_counts >= 50].index
    df_final = df_filtered_anime[df_filtered_anime['user_id'].isin(active_users)]
    print(f"Datos filtrados. Filas finales: {len(df_final)}")

    # Creación de la Matriz y Cálculo de Similitud
    user_item_matrix = df_final.pivot_table(index='user_id', columns='name', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    item_user_matrix = sparse_matrix.T
    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(item_similarity,
                                      index=user_item_matrix.columns,
                                      columns=user_item_matrix.columns)
    
    print("\n--- ¡Matriz de Similitud Generada! ---")

    # 3. GUARDAR la matriz recién generada
    print(f"Guardando la matriz de similitud en '{MODEL_FILE}'...")
    try:
        with open(MODEL_FILE, 'wb') as file:
            pickle.dump(item_similarity_df, file)
        print("¡Matriz guardada con éxito para la próxima vez!")
    except Exception as e:
        print(f"❌ Error al guardar la matriz de similitud: {e}")
        
# --- FIN DE LA LÓGICA DE CARGA Y ENTRENAMIENTO ---

if item_similarity_df is None:
    # Esto ocurre si el entrenamiento falló o si el script no pudo importar librerías
    print("\nNo se pudo obtener la Matriz de Similitud. El proceso de recomendación no puede continuar.")
    sys.exit(1)


## --- FUNCIÓN DE RECOMENDACIÓN (USO) ---
# Esta parte siempre se ejecuta, ya sea con la matriz cargada o recién creada.

def recommend_animes(anime_name, similarity_df, top_n=10):
    """
    Recomienda animes basados en la similitud de coseno con un anime dado.
    """
    # ... (Tu función original sin cambios)
    if anime_name not in similarity_df.index:
        return "Anime no encontrado en la matriz de similitud. Asegúrate de que el nombre sea exacto."

    similar_scores = similarity_df[anime_name]
    recommendations = similar_scores.sort_values(ascending=False)
    recommendations = recommendations.drop(labels=anime_name, errors='ignore')

    return recommendations.head(top_n)

# --- EJEMPLO de uso: ---
user_anime_input = input("\nIngresa el nombre de un anime que te guste (Ej: Death Note): ") 

if user_anime_input in item_similarity_df.index:
    print(f"\n--- Recomendaciones para {user_anime_input} ---")
    recommendations = recommend_animes(user_anime_input, item_similarity_df)
    print(recommendations)
else:
    print(f"El anime '{user_anime_input}' no está en el conjunto de animes populares/activos.")
    