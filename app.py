from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import requests
import logging
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar claves desde variables de entorno (hasta 20 claves)
API_KEYS_YOUTUBE = [
    os.environ.get(f'API_KEY_YOUTUBE_{i}') for i in range(1, 21)
]

# Filtrar las que estén vacías o no definidas
API_KEYS_YOUTUBE = [key for key in API_KEYS_YOUTUBE if key]

# Crear un iterador cíclico para rotar las claves automáticamente
api_key_iterator = itertools.cycle(API_KEYS_YOUTUBE)



# ------------------ FUNCIONES DE BASE DE DATOS ------------------
def conectar_postgres():
    """Conecta a la base de datos PostgreSQL"""
    try:
        return psycopg2.connect(
            host=os.environ.get('DB_HOST', 'dpg-d12hp249c44c738ar9qg-a.oregon-postgres.render.com'),
            database=os.environ.get('DB_NAME', 'bd_graututor'),
            user=os.environ.get('DB_USER', 'bd_graututor_user'),
            password=os.environ.get('DB_PASSWORD', 'vv19vRPRJzgptbk1sV4SzLPKZQbXqN42'),
            port=os.environ.get('DB_PORT', '5432')
        )
    except Exception as e:
        logger.error(f"Error conectando a la base de datos: {e}")
        raise

def ejecutar_consulta(query, params=None):
    """Ejecuta una consulta SQL y retorna un DataFrame"""
    conn = conectar_postgres()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        logger.error(f"Error ejecutando consulta: {e}")
        raise
    finally:
        conn.close()

# ------------------ FUNCIONES DE PROCESAMIENTO ------------------
def convertir_tiempo(t):
    """Convierte tiempo de formato MM:SS a minutos decimales"""
    if isinstance(t, str) and ':' in t:
        try:
            m, s = map(int, str(t).split(':'))
            return m + s / 60
        except:
            return float(t) if str(t).replace('.', '').isdigit() else 0
    else:
        return float(t) if str(t).replace('.', '').isdigit() else 0

def procesar_datos_estudiante(id_estudiante):
    """Procesa los datos de un estudiante específico"""
    try:
        # Consulta para obtener datos del estudiante
        query = "SELECT * FROM avance_estudiante WHERE id_estudiante = %s"
        df = ejecutar_consulta(query, (id_estudiante,))
        
        if df.empty:
            return None, f"No se encontraron datos para el estudiante {id_estudiante}"
        
        # Procesar fechas y tiempo
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['tiempo_min'] = df['tiempo_empleado'].apply(convertir_tiempo)
        df['hora'] = df['fecha'].dt.hour
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['fecha_dia'] = df['fecha'].dt.date
        
        # Calcular frecuencias por tema y día
        tema_dia_counts = df.groupby(['idexercise_type', 'fecha_dia']).size().reset_index(name='frecuencia_dia')
        frecuencia_total = tema_dia_counts.groupby('idexercise_type')['frecuencia_dia'].count().reset_index(name='dias_diferentes')
        
        # Agrupar datos por tipo de ejercicio
        df_grouped = df.groupby('idexercise_type').agg({
            'tiempo_min': 'mean',
            'errores_cometidos': ['mean', 'count', 'std'],
            'hora': 'mean',
            'dia_semana': 'mean'
        })
        
        df_grouped.columns = ['tiempo_promedio', 'errores_promedio', 'intentos', 'std_errores', 'hora_promedio', 'dia_promedio']
        df_grouped = df_grouped.reset_index()
        df_grouped = df_grouped.merge(frecuencia_total, on='idexercise_type', how='left')
        
        # Rellenar valores NaN
        df_grouped['std_errores'] = df_grouped['std_errores'].fillna(0)
        df_grouped['dias_diferentes'] = df_grouped['dias_diferentes'].fillna(1)
        
        # Calcular etiqueta de dificultad
        df_grouped['dificultad'] = (
            (df_grouped['tiempo_promedio'] > 30) |
            (df_grouped['errores_promedio'] > 1) |
            (df_grouped['intentos'] > 2)
        ).astype(int)
        
        return df_grouped, df
        
    except Exception as e:
        logger.error(f"Error procesando datos del estudiante {id_estudiante}: {e}")
        return None, str(e)

def entrenar_modelo(df_grouped):
    """Entrena el modelo Random Forest"""
    try:
        features = ['tiempo_promedio', 'errores_promedio', 'intentos']
        X = df_grouped[features].values
        y = df_grouped['dificultad'].values
        
        # Verificar que hay suficientes datos
        if len(X) < 2:
            return None, "Insuficientes datos para entrenar el modelo"
        
        # Modelo Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        rf_model.fit(X, y)
        y_pred_rf = rf_model.predict(X)
        
        # Obtener probabilidades
        proba_rf = rf_model.predict_proba(X)
        if proba_rf.shape[1] == 1:
            score_rf = proba_rf[:, 0]
        else:
            score_rf = proba_rf[:, 1]
            
        return rf_model, score_rf
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        return None, str(e)

def obtener_videos_youtube(tema_legible, max_results=2):
    """Obtiene videos de YouTube relacionados con el tema usando rotación de claves"""
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        api_key = next(api_key_iterator)  # Obtener la siguiente clave en la rotación

        params = {
            'part': 'snippet',
            'q': f"{tema_legible} matemáticas tutorial",
            'type': 'video',
            'maxResults': max_results,
            'key': api_key
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Fallo con clave: {api_key}, status: {response.status_code}")
            return []

        items = response.json().get('items', [])
        videos = []
        for item in items:
            snippet = item['snippet']
            video_id = item['id']['videoId']
            thumbnails = snippet.get('thumbnails', {})

            # Seleccionar la mejor calidad disponible
            thumbnail_url = thumbnails.get('maxres', {}).get('url') or \
                            thumbnails.get('high', {}).get('url') or \
                            thumbnails.get('medium', {}).get('url') or \
                            thumbnails.get('default', {}).get('url')

            videos.append({
                'titulo': snippet['title'],
                'descripcion': snippet['description'][:150] + '...' if len(snippet['description']) > 150 else snippet['description'],
                'canal': snippet['channelTitle'],
                'fecha_publicacion': snippet['publishedAt'],
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'video_id': video_id,
                'thumbnail_url': thumbnail_url,
                'thumbnail_embed': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            })

        return videos

    except Exception as e:
        logger.error(f"Error obteniendo videos de YouTube: {e}")
        return []

# ------------------ ENDPOINTS DE LA API ------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'API funcionando correctamente'
    })

@app.route('/student/<int:id_estudiante>/analysis', methods=['GET'])
def analyze_student(id_estudiante):
    """Analiza la dificultad de los ejercicios para un estudiante específico"""
    try:
        # Procesar datos del estudiante
        df_grouped, df_original = procesar_datos_estudiante(id_estudiante)
        
        if df_grouped is None:
            return jsonify({
                'error': df_original,
                'student_id': id_estudiante
            }), 404
        
        # Entrenar modelo
        model, scores = entrenar_modelo(df_grouped)
        
        if model is None:
            return jsonify({
                'error': scores,
                'student_id': id_estudiante
            }), 400
        
        # Asignar puntuaciones y niveles de dificultad
        df_grouped['score_dificultad_rf'] = scores
        df_grouped['nivel_dificultad_rf'] = pd.cut(
            df_grouped['score_dificultad_rf'],
            bins=[0, 0.4, 0.7, 1],
            labels=['Fácil', 'Intermedio', 'Difícil']
        )
        
        # Generar recomendaciones para temas difíciles
        recomendaciones = {}
        temas_dificiles = []
        
        for _, row in df_grouped.iterrows():
            tema_info = {
                'idexercise_type': row['idexercise_type'],
                'tiempo_promedio': round(row['tiempo_promedio'], 2),
                'errores_promedio': round(row['errores_promedio'], 2),
                'intentos': int(row['intentos']),
                'nivel_dificultad': str(row['nivel_dificultad_rf']),
                'score_dificultad': round(row['score_dificultad_rf'], 3)
            }
            
            # Obtener nombre legible del ejercicio
            ejercicio_nombre = df_original[df_original['idexercise_type'] == row['idexercise_type']]['ejercicio'].iloc[0]
            tema_info['ejercicio_nombre'] = ejercicio_nombre
            
            if row['nivel_dificultad_rf'] == 'Difícil':
                videos = obtener_videos_youtube(ejercicio_nombre)
                tema_info['videos_recomendados'] = videos
                temas_dificiles.append(tema_info)
            
            recomendaciones[row['idexercise_type']] = tema_info
        
        # Estadísticas generales
        total_ejercicios = len(df_grouped)
        ejercicios_faciles = len(df_grouped[df_grouped['nivel_dificultad_rf'] == 'Fácil'])
        ejercicios_intermedios = len(df_grouped[df_grouped['nivel_dificultad_rf'] == 'Intermedio'])
        ejercicios_dificiles = len(df_grouped[df_grouped['nivel_dificultad_rf'] == 'Difícil'])
        
        response = {
            'student_id': id_estudiante,
            'timestamp': datetime.now().isoformat(),
            'estadisticas_generales': {
                'total_tipos_ejercicios': total_ejercicios,
                'ejercicios_faciles': ejercicios_faciles,
                'ejercicios_intermedios': ejercicios_intermedios,
                'ejercicios_dificiles': ejercicios_dificiles,
                'porcentaje_dificiles': round((ejercicios_dificiles / total_ejercicios) * 100, 1) if total_ejercicios > 0 else 0
            },
            'analisis_detallado': recomendaciones,
            'temas_que_requieren_atencion': temas_dificiles,
            'total_registros_procesados': len(df_original)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analizando estudiante {id_estudiante}: {e}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}',
            'student_id': id_estudiante
        }), 500

@app.route('/student/<int:id_estudiante>/summary', methods=['GET'])
def student_summary(id_estudiante):
    """Resumen rápido del rendimiento del estudiante"""
    try:
        query = """
        SELECT 
            COUNT(*) as total_ejercicios,
            AVG(CASE WHEN tiempo_empleado ~ '^[0-9]+:[0-9]+$' 
                THEN EXTRACT(EPOCH FROM tiempo_empleado::interval)/60 
                ELSE tiempo_empleado::float END) as tiempo_promedio_general,
            AVG(errores_cometidos) as errores_promedio_general,
            COUNT(DISTINCT idexercise_type) as tipos_ejercicios_diferentes,
            MAX(fecha) as ultima_actividad
        FROM avance_estudiante 
        WHERE id_estudiante = %s
        """
        
        df_summary = ejecutar_consulta(query, (id_estudiante,))
        
        if df_summary.empty or df_summary.iloc[0]['total_ejercicios'] == 0:
            return jsonify({
                'error': f'No se encontraron datos para el estudiante {id_estudiante}',
                'student_id': id_estudiante
            }), 404
        
        summary = df_summary.iloc[0]
        
        return jsonify({
            'student_id': id_estudiante,
            'resumen': {
                'total_ejercicios_realizados': int(summary['total_ejercicios']),
                'tiempo_promedio_minutos': round(float(summary['tiempo_promedio_general'] or 0), 2),
                'errores_promedio': round(float(summary['errores_promedio_general'] or 0), 2),
                'tipos_ejercicios_diferentes': int(summary['tipos_ejercicios_diferentes']),
                'ultima_actividad': summary['ultima_actividad'].isoformat() if summary['ultima_actividad'] else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo resumen del estudiante {id_estudiante}: {e}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}',
            'student_id': id_estudiante
        }), 500

@app.route('/student/<int:id_estudiante>/recommendations', methods=['GET'])
def get_student_recommendations(id_estudiante):
    """Obtiene solo las recomendaciones de videos para temas difíciles del estudiante"""
    try:
        # Obtener parámetros opcionales
        max_videos = request.args.get('max_videos', 3, type=int)
        
        # Procesar datos del estudiante
        df_grouped, df_original = procesar_datos_estudiante(id_estudiante)
        
        if df_grouped is None:
            return jsonify({
                'error': df_original,
                'student_id': id_estudiante
            }), 404
        
        # Entrenar modelo
        model, scores = entrenar_modelo(df_grouped)
        
        if model is None:
            return jsonify({
                'error': scores,
                'student_id': id_estudiante
            }), 400
        
        # Asignar puntuaciones y niveles de dificultad
        df_grouped['score_dificultad_rf'] = scores
        df_grouped['nivel_dificultad_rf'] = pd.cut(
            df_grouped['score_dificultad_rf'],
            bins=[0, 0.4, 0.7, 1],
            labels=['Fácil', 'Intermedio', 'Difícil']
        )
        
        # Generar solo recomendaciones para temas difíciles
        recomendaciones = []
        
        for _, row in df_grouped.iterrows():
            if row['nivel_dificultad_rf'] == 'Difícil':
                ejercicio_nombre = df_original[df_original['idexercise_type'] == row['idexercise_type']]['ejercicio'].iloc[0]
                videos = obtener_videos_youtube(ejercicio_nombre, max_videos)
                
                tema_info = {
                    'idexercise_type': row['idexercise_type'],
                    'ejercicio_nombre': ejercicio_nombre,
                    'tiempo_promedio': round(row['tiempo_promedio'], 2),
                    'errores_promedio': round(row['errores_promedio'], 2),
                    'intentos': int(row['intentos']),
                    'score_dificultad': round(row['score_dificultad_rf'], 3),
                    'videos_recomendados': videos,
                    'total_videos_encontrados': len(videos)
                }
                recomendaciones.append(tema_info)
        
        return jsonify({
            'student_id': id_estudiante,
            'timestamp': datetime.now().isoformat(),
            'total_temas_dificiles': len(recomendaciones),
            'recomendaciones': recomendaciones
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo recomendaciones para estudiante {id_estudiante}: {e}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}',
            'student_id': id_estudiante
        }), 500

@app.route('/students', methods=['GET'])
def list_students():
    """Lista todos los estudiantes disponibles"""
    try:
        query = """
        SELECT DISTINCT id_estudiante, 
               COUNT(*) as total_ejercicios,
               MAX(fecha) as ultima_actividad
        FROM avance_estudiante 
        GROUP BY id_estudiante 
        ORDER BY id_estudiante
        """
        
        df_students = ejecutar_consulta(query)
        
        students = []
        for _, row in df_students.iterrows():
            students.append({
                'id_estudiante': int(row['id_estudiante']),
                'total_ejercicios': int(row['total_ejercicios']),
                'ultima_actividad': row['ultima_actividad'].isoformat() if row['ultima_actividad'] else None
            })
        
        return jsonify({
            'total_estudiantes': len(students),
            'estudiantes': students
        })
        
    except Exception as e:
        logger.error(f"Error listando estudiantes: {e}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
