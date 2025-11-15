# 🍜 Sistema de Recomendación de Anime (Filtrado Colaborativo basado en Ítems)

Este proyecto implementa un **Sistema de Recomendación de Anime** utilizando la técnica de **Filtrado Colaborativo basado en Ítems (Item-Based Collaborative Filtering)**. El objetivo es sugerir animes similares a uno dado, basándose en los patrones de calificación de los usuarios.

---

## 🎯 Objetivo del Proyecto

El proyecto se centra en:
1.  **Cargar y preprocesar** un conjunto de datos masivo de calificaciones de anime.
2.  **Filtrar** los datos para incluir solo animes populares y usuarios activos, garantizando la calidad de las recomendaciones.
3.  **Construir una Matriz de Interacción Usuario-Ítem**.
4.  **Calcular la similitud de coseno** entre los animes (ítems) para determinar cuáles son más parecidos.
5.  **Generar recomendaciones** para cualquier anime de la matriz.

---

## ⚙️ Tecnologías Utilizadas

* **Python**
* **Pandas**: Para la manipulación y preprocesamiento de los DataFrames.
* **KaggleHub**: Para la descarga programática del dataset.
* **SciPy (CSR Matrix)**: Para la optimización de la matriz de interacción dispersa.
* **Scikit-learn (`cosine_similarity`)**: Para el cálculo eficiente de la similitud de coseno.

---

## 💾 Dataset

El proyecto utiliza el conjunto de datos de **Anime Recommendations Database** de Kaggle.

* **Fuente:** `CooperUnion/anime-recommendations-database` (Descargado vía `kagglehub`).
* **Archivos clave:**
    * `rating.csv`: Contiene las calificaciones de los usuarios.
    * `anime.csv`: Contiene los metadatos del anime (incluyendo el nombre).

---

## 💡 Metodología (Flujo de Trabajo del Código)

### 1. Preparación de Datos y Carga
* Los archivos `rating.csv` y `anime.csv` son descargados y cargados en DataFrames de Pandas.
* **Limpieza de Datos:** Las calificaciones de **-1** (que significan que el usuario solo vio, pero no calificó) se eliminan del DataFrame de calificaciones.

### 2. Fusión y Filtrado
* Los DataFrames de calificaciones y animes se combinan (`inner merge`) usando el `anime_id`.
* **Filtrado de Animes (Popularidad):** Se eliminan los animes con **menos de 50 calificaciones**, ya que no ofrecen suficiente información para una similitud confiable.
* **Filtrado de Usuarios (Actividad):** Se eliminan los usuarios que han calificado **menos de 50 animes** en el conjunto filtrado, para enfocarse en usuarios con un historial de interacción significativo.

### 3. Creación de la Matriz de Interacción
* Se utiliza la función `pivot_table` de Pandas para crear la **Matriz Usuario-Ítem**, donde:
    * **Índice:** `user_id`
    * **Columnas:** `name` del Anime
    * **Valores:** `rating`
* Los valores `NaN` (ausencia de calificación) se rellenan con **0** antes del cálculo de similitud.
* La matriz se convierte a un formato **CSR (Compressed Sparse Row) de SciPy** para optimizar el rendimiento y el uso de memoria.

### 4. Cálculo de Similitud
* La matriz dispersa se **transpone** (convirtiéndola a Matriz Ítem-Usuario).
* Se aplica el algoritmo de **similitud de coseno** sobre la matriz transpuesta. Esto genera una Matriz de Similitud donde cada valor $[i, j]$ representa qué tan similares son el Anime $i$ y el Anime $j$.

### 5. Función de Recomendación
* Se define la función `recommend_animes(anime_name, similarity_df, top_n=10)`:
    * Busca la fila del `anime_name` en la Matriz de Similitud.
    * Ordena las puntuaciones de similitud de forma descendente.
    * Excluye el propio anime (ya que siempre tendrá una similitud de 1.0).
    * Devuelve los **top N** animes más similares.

---

## 🚀 Uso (Ejemplo)

Para obtener las recomendaciones, simplemente llama a la función `recommend_animes` con el nombre exacto de un anime que esté en la matriz final.

```python
# Ejemplo
anime_ejemplo = 'Cowboy Bebop'
recommendations = recommend_animes(anime_ejemplo, item_similarity_df)
print(f"--- Recomendaciones para {anime_ejemplo} ---")
print(recommendations) 
```

## 🛠️ Instalación y Ejecución

Para poner en marcha este sistema de recomendación, sigue estos pasos:

### 1. Requisitos Previos

Asegúrate de tener instalado **Python** (versión 3.6 o superior) y la herramienta de línea de comandos de **Kaggle** configurada para la descarga del dataset.

### 2. Instalación de Dependencias

Instala todas las librerías necesarias utilizando `pip`:

```bash
pip install pandas kagglehub scikit-learn scipy
python My_anime_recomendator_FC.py
```