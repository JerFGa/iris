# 🌸 Clasificador IRIS Pedagógico

Aplicación interactiva construida con **Streamlit** y **scikit-learn** para la exploración pedagógica, análisis exploratorio de datos (EDA) y clasificación multiclase sobre el dataset clásico **Fisher's Iris** (*Setosa, Versicolor, Virginica*).

## 🚀 Características
- **Visualización y EDA Interactivo:**
  - Gráficos de dispersión 2D y 3D interactivos con Plotly
  - Análisis de correlación y distribuciones univariadas/bivariadas por especie
- **Modelos de Aprendizaje Supervisado:**
  - K-Nearest Neighbors (KNN)
  - Árbol de Decisión con visualización gráfica de reglas
  - Support Vector Machines (SVM con kernel lineal y RBF)
  - Random Forest, Naive Bayes y Regresión Logística
- **Simulador de Predicción en Vivo:**
  - Controles deslizantes (sliders) interactivos para medidas de sépalo y pétalo
  - Inferencia en tiempo real con distribución porcentual de probabilidades por especie
  - Pruebas masivas aleatorias con validación de aciertos en vivo

## 🛠️ Tecnologías
- **Python 3**
- **Streamlit**
- **scikit-learn**, **Pandas**, **NumPy**
- **Plotly**, **Seaborn**, **Matplotlib**

## 📦 Instalación y Ejecución
```bash
pip install -r requirements.txt
streamlit run main.py
```
