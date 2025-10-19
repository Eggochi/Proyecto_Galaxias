# Proyecto_Galaxias

Proyecto de curso: Aprendizaje computacional  
Objetivo: Aprendizaje semi-supervisado / clasificación de galaxias usando aprendizaje semi-supervisado.

## Resumen
Entrenamiento y evaluación de un modelo (EfficientNet-B0 por defecto) sobre un dataset de imágenes de galaxias. Soporta:
- Cargar datos desde un CSV (primera columna = path/filename, última columna = clase por defecto).
- Datasets etiquetados y no etiquetados (inferencia).
- Early stopping y guardado del mejor modelo.
- CSV-based Dataset para evitar mover archivos a carpetas por clase.