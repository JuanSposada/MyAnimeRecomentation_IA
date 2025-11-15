import pandas as pd
import kagglehub

# La ruta donde se descargó el dataset (esto ya lo ejecutaste)
dataset_path = kagglehub.dataset_download("CooperUnion/anime-recommendations-database")
print("Dataset descargado en:", dataset_path)

# Construir las rutas completas a los archivos
ratings_file = f"{dataset_path}/rating.csv"
anime_file = f"{dataset_path}/anime.csv"

# Cargar los DataFrames
df_ratings = pd.read_csv(ratings_file)
df_anime = pd.read_csv(anime_file)

print("DataFrame de Calificaciones (df_ratings) cargado con", len(df_ratings), "filas.")
print("DataFrame de Anime (df_anime) cargado con", len(df_anime), "filas.")

# Eliminar las filas donde el rating es -1
df_ratings_clean = df_ratings[df_ratings['rating'] != -1]

print("Filas eliminadas (rating = -1):", len(df_ratings) - len(df_ratings_clean))
print("Filas restantes para el modelo:", len(df_ratings_clean))

print("\n--- Vista previa de df_ratings_clean ---")
print(df_ratings_clean.head())

print("\n--- Información de df_anime ---")
print(df_anime.info())

# Combinar los DataFrames
# Usamos un 'inner' merge para asegurar que solo incluimos animes que tienen calificaciones y viceversa.
df_merged = pd.merge(df_ratings_clean,
                     df_anime[['anime_id', 'name']], # Solo necesitamos 'anime_id' y 'name' del df_anime
                     on='anime_id',
                     how='inner')

print(f"DataFrame Combinado (df_merged) creado con {len(df_merged)} filas.")
print("\n--- Vista previa de df_merged ---")
print(df_merged.head())

# Contar cuántas calificaciones tiene cada anime (por nombre)
anime_rating_counts = df_merged.groupby('name')['rating'].count()

# Identificar los animes que cumplen el umbral (e.g., al menos 50 calificaciones)
popular_animes = anime_rating_counts[anime_rating_counts >= 50].index

# Filtrar el DataFrame combinado para incluir solo los animes populares
df_filtered_anime = df_merged[df_merged['name'].isin(popular_animes)]

print(f"Animes restantes después del filtro (min 50 calificaciones): {len(popular_animes)} de {len(anime_rating_counts)}")
print(f"Filas restantes en el DF: {len(df_filtered_anime)}")

# Contar cuántas calificaciones ha dado cada usuario
user_rating_counts = df_filtered_anime.groupby('user_id')['rating'].count()

# Identificar los usuarios que cumplen el umbral (e.g., al menos 50 calificaciones)
active_users = user_rating_counts[user_rating_counts >= 50].index

# Filtrar el DataFrame final
df_final = df_filtered_anime[df_filtered_anime['user_id'].isin(active_users)]

print(f"Usuarios restantes después del filtro (min 50 calificaciones): {len(active_users)}")
print(f"Filas finales listas para modelar: {len(df_final)}")

# Crear la matriz Usuario-Ítem (Matriz de Interacción)
user_item_matrix = df_final.pivot_table(
    index='user_id',
    columns='name',
    values='rating'
)

print("\n--- Vista previa de la Matriz de Interacción (User-Item Matrix) ---")
print(user_item_matrix.head())
print(f"Tamaño de la Matriz: {user_item_matrix.shape}")

from scipy.sparse import csr_matrix

# Rellenar los NaN con 0. Esto es común antes de aplicar la similitud de coseno,
# pero OJO: el 0 aquí significa "sin calificación", NO "calificación de 0".
matrix_for_similarity = user_item_matrix.fillna(0)

# Convertir la matriz de Pandas a una matriz dispersa (CSR Matrix)
# Esto reduce el consumo de memoria y acelera los cálculos.
sparse_matrix = csr_matrix(matrix_for_similarity.values)

from sklearn.metrics.pairwise import cosine_similarity

# 1. Transponer la matriz para calcular la similitud entre columnas (Animes)
# La matriz debe ser Ítem-Usuario (Anime-Usuario)
item_user_matrix = sparse_matrix.T

# 2. Calcular la similitud de coseno
# Esto resulta en una matriz (5172 x 5172) donde cada celda [i, j] es la similitud entre el Anime i y el Anime j.
item_similarity = cosine_similarity(item_user_matrix)

# 3. Convertir la matriz de similitud de vuelta a un DataFrame para fácil acceso (opcional)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=user_item_matrix.columns,
                                  columns=user_item_matrix.columns)

print("\n--- Vista previa de la Matriz de Similitud entre Animes ---")
print(item_similarity_df.head())


def recommend_animes(anime_name, similarity_df, top_n=10):
    """
    Recomienda animes basados en la similitud de coseno con un anime dado.
    """
    if anime_name not in similarity_df.index:
        return "Anime no encontrado en la matriz de similitud. Asegúrate de que el nombre sea exacto."

    # Obtener las puntuaciones de similitud para el anime_name
    similar_scores = similarity_df[anime_name]

    # Ordenar los animes por puntuación de similitud de forma descendente
    # Excluir el propio anime_name (similitud 1.0)
    recommendations = similar_scores.sort_values(ascending=False)
    recommendations = recommendations.drop(labels=anime_name, errors='ignore')

    # Devolver los N principales
    return recommendations.head(top_n)

# EJEMPLO de uso:
# Asegúrate de usar un nombre de anime EXACTO que esté en las columnas de user_item_matrix
anime_ejemplo = 'Cowboy Bebop' # Reemplaza con un nombre de tu matriz
if anime_ejemplo in item_similarity_df.index:
    print(f"\n--- Recomendaciones para {anime_ejemplo} ---")
    recommendations = recommend_animes(anime_ejemplo, item_similarity_df)
    print(recommendations)
else:
    print(f"El anime '{anime_ejemplo}' no está en el conjunto de animes populares.")