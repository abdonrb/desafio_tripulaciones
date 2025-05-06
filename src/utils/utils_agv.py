# Versión 1 preparación para Machine Learning
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

def df_to_ml(df_original, max_tfidf_features=5):
    """
    Prepara un DataFrame para el entrenamiento de un modelo de recomendación KNN.
    
    Parámetros:
    -----------
    df_original : DataFrame
        El DataFrame original que contiene la información de las actividades.
    max_tfidf_features : int, opcional (default=5)
        Número máximo de características a extraer con TF-IDF.
        
    Retorna:
    --------
    DataFrame preparado para machine learning con todas las características numéricas.
    """
    # Crear una copia del DataFrame original para no modificarlo
    df_ml = df_original.copy()
    
    # 1. Limpieza básica de datos
    # ---------------------------
    df_ml.replace(["No especificada", "No especificado", "No disponible"], np.nan, inplace=True)
    
    # 2. Procesamiento de texto con TF-IDF
    # -----------------------------------
    # Extraer longitud del título y descripción
    df_ml['titulo_longitud'] = df_ml['título'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df_ml['descripcion_longitud'] = df_ml['descripción'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Definir stop words en español
    stop_words = ['de', 'la', 'el', 'en', 'y', 'a', 'para', 'con', 'por', 'del', 
                  'los', 'las', 'un', 'una', 'al', 'es', 'su', 'se', 'que', 'no']
    
    # TF-IDF para título
    if df_ml['título'].notna().sum() > 1:
        tfidf_titulo = TfidfVectorizer(
            max_features=max_tfidf_features, 
            stop_words=stop_words,
            ngram_range=(1, 2)  # Incluir bigramas para capturar frases clave
        )
        titulo_tfidf = tfidf_titulo.fit_transform(df_ml['título'].fillna(''))
        titulo_tfidf_df = pd.DataFrame(
            titulo_tfidf.toarray(), 
            columns=[f'titulo_tfidf_{i}' for i in range(titulo_tfidf.shape[1])],
            index=df_ml.index
        )
        df_ml = pd.concat([df_ml, titulo_tfidf_df], axis=1)
    
    # TF-IDF para descripción
    if df_ml['descripción'].notna().sum() > 1:
        tfidf_desc = TfidfVectorizer(
            max_features=max_tfidf_features, 
            stop_words=stop_words,
            ngram_range=(1, 2)
        )
        desc_tfidf = tfidf_desc.fit_transform(df_ml['descripción'].fillna(''))
        desc_tfidf_df = pd.DataFrame(
            desc_tfidf.toarray(), 
            columns=[f'desc_tfidf_{i}' for i in range(desc_tfidf.shape[1])],
            index=df_ml.index
        )
        df_ml = pd.concat([df_ml, desc_tfidf_df], axis=1)
    
    # 3. Procesamiento de columnas categóricas
    # --------------------------------------
    
    # Procesar categoría y subcategoría con OneHotEncoder
    categorical_cols = ['categoría', 'subcategoría']
    for col in categorical_cols:
        if col in df_ml.columns:
            # Convertir a string para asegurar compatibilidad
            df_ml[col] = df_ml[col].astype(str)
            
            # Manejar múltiples categorías (separadas por puntos)
            # Expandir columnas múltiples en columnas individuales
            categories = set()
            for val in df_ml[col].dropna():
                for category in val.split('. '):
                    categories.add(category)
            
            # Crear columnas binarias para cada categoría
            for category in categories:
                if category and category != 'nan':
                    col_name = f"{col}_{category}"
                    df_ml[col_name] = df_ml[col].apply(
                        lambda x: 1 if pd.notna(x) and category in x else 0
                    )
    
    # 4. Procesar edad (si no está ya procesada)
    # ----------------------------------------
    if 'edad_min' in df_ml.columns and 'edad_max' in df_ml.columns:
        # Rellenar valores faltantes con valores por defecto
        df_ml['edad_min'] = pd.to_numeric(df_ml['edad_min'], errors='coerce').fillna(0)
        df_ml['edad_max'] = pd.to_numeric(df_ml['edad_max'], errors='coerce').fillna(18)
        
        # Crear rangos de edad como características
        age_ranges = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18)]
        for min_age, max_age in age_ranges:
            col_name = f"edad_{min_age}_{max_age}"
            df_ml[col_name] = (
                ((df_ml['edad_min'] <= max_age) & (df_ml['edad_max'] >= min_age)) | 
                ((df_ml['edad_min'] <= min_age) & (df_ml['edad_max'] >= max_age))
            ).astype(int)
    
    # 5. Procesar fechas
    # ----------------
    date_cols = ['fecha_inicio', 'fecha_fin']
    for col in date_cols:
        if col in df_ml.columns:
            # Convertir a datetime si no lo está ya
            if not pd.api.types.is_datetime64_any_dtype(df_ml[col]):
                df_ml[col] = pd.to_datetime(df_ml[col], errors='coerce', dayfirst=True)
            
            if df_ml[col].notna().any():
                # Extraer componentes de fecha
                df_ml[f'{col}_year'] = df_ml[col].dt.year
                df_ml[f'{col}_month'] = df_ml[col].dt.month
                df_ml[f'{col}_day'] = df_ml[col].dt.day
                df_ml[f'{col}_dayofweek'] = df_ml[col].dt.dayofweek
                
                # Características estacionales
                df_ml[f'{col}_is_summer'] = ((df_ml[col].dt.month >= 6) & 
                                             (df_ml[col].dt.month <= 9)).astype(int)
                df_ml[f'{col}_is_winter'] = ((df_ml[col].dt.month == 12) | 
                                             (df_ml[col].dt.month <= 2)).astype(int)
                df_ml[f'{col}_is_weekend'] = (df_ml[col].dt.dayofweek >= 5).astype(int)
    
    # Calcular duración del evento en días
    if 'fecha_inicio' in df_ml.columns and 'fecha_fin' in df_ml.columns:
        if pd.api.types.is_datetime64_any_dtype(df_ml['fecha_inicio']) and pd.api.types.is_datetime64_any_dtype(df_ml['fecha_fin']):
            df_ml['duracion_dias'] = (df_ml['fecha_fin'] - df_ml['fecha_inicio']).dt.days + 1
            df_ml['duracion_dias'] = df_ml['duracion_dias'].fillna(1)  # Asumimos 1 día para eventos sin rango
            
            # Codificar duración en rangos
            df_ml['duracion_corta'] = (df_ml['duracion_dias'] <= 1).astype(int)
            df_ml['duracion_media'] = ((df_ml['duracion_dias'] > 1) & 
                                      (df_ml['duracion_dias'] <= 7)).astype(int)
            df_ml['duracion_larga'] = (df_ml['duracion_dias'] > 7).astype(int)
    
    # 6. Procesar horas
    # ---------------
    time_cols = ['hora_inicio', 'hora_fin']
    for col in time_cols:
        if col in df_ml.columns:
            # Convertir a formato numérico si hay datos(horas decimales)
            df_ml[col] = pd.to_datetime(df_ml[col], format='%H:%M', errors='coerce')
            df_ml[col] = df_ml[col].dt.hour + df_ml[col].dt.minute / 60
            
            # Características de periodo del día
            if pd.notna(df_ml[col]).any():
                df_ml[f'{col}_manana'] = ((df_ml[col] >= 7) & (df_ml[col] < 12)).astype(int)
                df_ml[f'{col}_mediodia'] = ((df_ml[col] >= 12) & (df_ml[col] < 15)).astype(int)
                df_ml[f'{col}_tarde'] = ((df_ml[col] >= 15) & (df_ml[col] < 19)).astype(int)
                df_ml[f'{col}_noche'] = ((df_ml[col] >= 19) | (df_ml[col] < 7)).astype(int)
    
    # 7. Procesar días de la semana
    # --------------------------
    if 'día_días' in df_ml.columns:
        dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        for dia in dias_semana:
            df_ml[f'dia_{dia}'] = df_ml['día_días'].str.contains(dia, case=False, na=False).astype(int)
        
        # Agregar características de fin de semana y entre semana
        df_ml['dia_findesemana'] = ((df_ml['dia_sábado'] == 1) | (df_ml['dia_domingo'] == 1)).astype(int)
        df_ml['dia_entresemana'] = ((df_ml['dia_lunes'] == 1) | (df_ml['dia_martes'] == 1) | 
                                   (df_ml['dia_miércoles'] == 1) | (df_ml['dia_jueves'] == 1) | 
                                   (df_ml['dia_viernes'] == 1)).astype(int)
    
     # 8. Procesar periodicidad
    # ---------------------
    if 'periodicidad' in df_ml.columns:
        df_ml['periodicidad'] = df_ml['periodicidad'].fillna('Ninguna')
        df_ml['periodicidad_diaria'] = (df_ml['periodicidad'] == 'diaria').astype(int)
        df_ml['periodicidad_semanal'] = (df_ml['periodicidad'] == 'semanal').astype(int)
        df_ml['periodicidad_quincenal'] = (df_ml['periodicidad'] == 'quincenal').astype(int)
        df_ml['periodicidad_mensual'] = (df_ml['periodicidad'] == 'mensual').astype(int)
    
    
    # 9. Procesar distritos
    # ------------------
    if 'distrito' in df_ml.columns:
        df_ml['distrito'] = df_ml['distrito'].fillna('No especificado')
        distritos = df_ml['distrito'].unique()
        
        distrito_cols = {
            f'distrito_{d}': (df_ml['distrito'] == d).astype(int) 
            for d in distritos if d and d != 'nan'
        }
    
        df_ml = pd.concat([df_ml, pd.DataFrame(distrito_cols, index=df_ml.index)], axis=1)

    
    # 10. Procesar otras características booleanas
    # ---------------------------------------
    # Convertir columnas booleanas a numéricas
    bool_cols = ['requiere_inscripcion', 'recomendado', 'tiene_url_info', 'tiene_url_imagen']
    for col in bool_cols:
        if col in df_ml.columns:
            # Asegurarse de que pandas infiera el tipo de la columna antes de trabajar con ella
            # df_ml[col] = df_ml[col].infer_objects(copy=False)
            df_ml[col] = df_ml[col].astype('boolean').fillna(False)  # Convertir a booleano y llenar NaN
            df_ml[col] = df_ml[col].astype(int)  # Convertir a entero
    
    # 11. Procesar precio
    # ----------------
    if 'precio_valor' in df_ml.columns:
        df_ml['precio_valor'] = pd.to_numeric(df_ml['precio_valor'], errors='coerce')
        df_ml['precio_gratuito'] = (df_ml['precio_valor'] == 0).fillna(False).astype(int)
        df_ml['precio_bajo'] = ((df_ml['precio_valor'] > 0) & (df_ml['precio_valor'] <= 10)).fillna(False).astype(int)
        df_ml['precio_medio'] = ((df_ml['precio_valor'] > 10) & (df_ml['precio_valor'] <= 30)).fillna(False).astype(int)
        df_ml['precio_alto'] = (df_ml['precio_valor'] > 30).fillna(False).astype(int)
    
    # 12. Agregar columna ciudad_madrid para uniformidad
    # ----------------------------------------------
    df_ml['ciudad_Madrid'] = 1
    
    # 13. Eliminar columnas originales que ya han sido procesadas
    # ------------------------------------------------------
    columnas_a_eliminar = [
        'título', 'descripción', 'categoría', 'subcategoría', 'ciudad',
        'día_días', 'distrito', 'fecha_inicio', 'fecha_fin', 'periodicidad',
    ]
    cols_to_drop = [col for col in columnas_a_eliminar if col in df_ml.columns]
    df_ml = df_ml.drop(columns=cols_to_drop)
    
    # 14. Opcional: Estandarización de variables numéricas
    # ------------------------------------------------
    # Identificar columnas numéricas para estandarización
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
    
    # No estandarizamos columnas binarias (0/1)
    binary_cols = []
    for col in numeric_cols:
        if set(df_ml[col].dropna().unique()).issubset({0, 1}):
            binary_cols.append(col)
    
    # Columnas a estandarizar (numéricas que no son binarias)
    scale_cols = [col for col in numeric_cols if col not in binary_cols]
    
    # Aplicar StandardScaler a estas columnas
    if scale_cols:
        scaler = StandardScaler()
        df_ml[scale_cols] = scaler.fit_transform(df_ml[scale_cols])
    
    # 15. Manejo de valores faltantes finales
    # ------------------------------------
    # Identificar columnas numéricas
    cols_numericas = df_ml.select_dtypes(include=['number']).columns

    # Rellenar NaN solo en esas columnas
    df_ml[cols_numericas] = df_ml[cols_numericas].fillna(0)


    
    return df_ml