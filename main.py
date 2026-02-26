"""
🌸 Clasificador IRIS Pedagógico
================================
Aplicación interactiva para aprender y explorar clasificación con el dataset IRIS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador IRIS",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# ESTILOS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        border-radius: 12px; padding: 1rem; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #e8f4f8; border-left: 4px solid #2196F3;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        border-radius: 16px; padding: 2rem; text-align: center;
        font-size: 1.5rem; font-weight: 700; margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATOS & CACHE
# ──────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["especie"] = [iris.target_names[t] for t in iris.target]
    df["target"] = iris.target
    return df, iris

@st.cache_resource
def entrenar_modelo(algoritmo, test_size, random_state, escalar, **params):
    df, iris = cargar_datos()
    X = df[iris.feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if escalar:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    modelos = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(**{k: v for k, v in params.items() if k in ["n_neighbors", "metric"]}),
        "Árbol de Decisión": DecisionTreeClassifier(**{k: v for k, v in params.items() if k in ["max_depth", "criterion"]}),
        "SVM": SVC(**{k: v for k, v in params.items() if k in ["C", "kernel"]}, probability=True),
        "Naive Bayes": GaussianNB(),
        "Regresión Logística": LogisticRegression(max_iter=500, **{k: v for k, v in params.items() if k in ["C"]}),
        "Random Forest": RandomForestClassifier(**{k: v for k, v in params.items() if k in ["n_estimators", "max_depth"]}, random_state=random_state),
    }
    modelo = modelos[algoritmo]
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return modelo, scaler, X_train, X_test, y_train, y_test, y_pred, iris.feature_names, iris.target_names


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 📊 Datos")
    test_size = st.slider("Tamaño del conjunto de prueba", 0.10, 0.40, 0.20, 0.05,
                          help="Fracción del dataset usado para evaluar el modelo")
    random_state = st.number_input("Semilla aleatoria", 0, 999, 42,
                                   help="Controla la reproducibilidad del experimento")
    escalar = st.checkbox("Escalar características (StandardScaler)", value=True,
                          help="Normaliza los datos. Recomendado para KNN y SVM")

    st.markdown("### 🤖 Algoritmo")
    algoritmo = st.selectbox("Clasificador", [
        "K-Nearest Neighbors (KNN)",
        "Árbol de Decisión",
        "SVM",
        "Naive Bayes",
        "Regresión Logística",
        "Random Forest",
    ])

    st.markdown("### 🔧 Hiperparámetros")
    params = {}
    if algoritmo == "K-Nearest Neighbors (KNN)":
        params["n_neighbors"] = st.slider("Número de vecinos (k)", 1, 20, 5)
        params["metric"] = st.selectbox("Métrica de distancia", ["euclidean", "manhattan", "minkowski"])
    elif algoritmo == "Árbol de Decisión":
        params["max_depth"] = st.slider("Profundidad máxima", 1, 10, 3)
        params["criterion"] = st.selectbox("Criterio de división", ["gini", "entropy"])
    elif algoritmo == "SVM":
        params["C"] = st.slider("Parámetro C (regularización)", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    elif algoritmo == "Regresión Logística":
        params["C"] = st.slider("Parámetro C", 0.01, 10.0, 1.0, 0.01)
    elif algoritmo == "Random Forest":
        params["n_estimators"] = st.slider("Número de árboles", 10, 200, 100, 10)
        params["max_depth"] = st.slider("Profundidad máxima", 1, 10, 5)

    st.markdown("---")
    st.markdown("🌸 **IRIS Classifier** · Pedagógico")


# ──────────────────────────────────────────────
# TITULO
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Clasificador IRIS Interactivo</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; font-size:1.1rem;'>Aprende Machine Learning explorando el dataset más famoso del mundo</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ENTRENAR
# ──────────────────────────────────────────────
with st.spinner("⚙️ Entrenando modelo..."):
    resultado = entrenar_modelo(algoritmo, test_size, random_state, escalar, **params)
modelo, scaler, X_train, X_test, y_train, y_test, y_pred, feature_names, target_names = resultado
df, iris = cargar_datos()

accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(modelo, 
    StandardScaler().fit_transform(df[iris.feature_names].values) if escalar else df[iris.feature_names].values,
    df["target"].values, cv=5)

# ──────────────────────────────────────────────
# MÉTRICAS RÁPIDAS
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Exactitud (Test)", f"{accuracy*100:.1f}%")
col2.metric("📈 CV Media (5-fold)", f"{cv_scores.mean()*100:.1f}%")
col3.metric("📉 CV Desv. Estándar", f"±{cv_scores.std()*100:.1f}%")
col4.metric("🔢 Muestras de Prueba", len(y_test))

st.markdown("---")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Explorar Datos",
    "📊 Desempeño del Modelo",
    "🔍 Análisis Visual",
    "🌳 Interpretabilidad",
    "🧪 Predecir"
])


# ══════════════════════════════════════════════
# TAB 1 — EXPLORAR DATOS
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## 📚 El Dataset IRIS")
    st.markdown("""
    <div class="info-box">
    El dataset <b>IRIS</b> fue introducido por Ronald Fisher en 1936. Contiene 150 muestras de 3 especies 
    de iris (<i>Setosa, Versicolor, Virginica</i>), con 4 características medidas en centímetros: 
    longitud y ancho del sépalo y pétalo. Es el <i>"Hola Mundo"</i> del Machine Learning.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.sample(20, random_state=1).reset_index(drop=True), use_container_width=True)
    with col2:
        st.markdown("#### 📐 Estadísticas")
        st.dataframe(df.drop(columns=["target"]).describe().round(2), use_container_width=True)

    st.markdown("#### 🔵 Distribución de Especies")
    fig = px.histogram(df, x="especie", color="especie",
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       title="Distribución de clases (50 muestras por especie)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📦 Boxplots por Característica y Especie")
    fig = make_subplots(rows=2, cols=2, subplot_titles=feature_names)
    colors = {"setosa": "#FF9999", "versicolor": "#66B2FF", "virginica": "#99FF99"}
    for i, feat in enumerate(feature_names):
        row, col = divmod(i, 2)
        for especie in target_names:
            vals = df[df["especie"] == especie][feat]
            fig.add_trace(go.Box(y=vals, name=especie, marker_color=colors[especie],
                                 showlegend=(i == 0)), row=row+1, col=col+1)
    fig.update_layout(height=500, title_text="Distribución de características por especie")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔗 Matriz de Correlación")
    corr = df[feature_names].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Correlación entre características")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — DESEMPEÑO
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f"## 📊 Desempeño: {algoritmo}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True,
                        x=target_names, y=target_names,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicción", y="Real", color="Conteo"),
                        title="Matriz de Confusión")
        fig.update_xaxes(title="Predicción")
        fig.update_yaxes(title="Valor Real")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Reporte de Clasificación")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df.style.background_gradient(cmap="Greens", subset=["precision", "recall", "f1-score"]),
                     use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>Precisión:</b> de los que predije como X, ¿cuántos eran X?<br>
        <b>Recall:</b> de los que eran X, ¿cuántos detecté?<br>
        <b>F1-Score:</b> media armónica de precisión y recall
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📈 Validación Cruzada (5-Fold)")
    fig = go.Figure()
    fig.add_bar(x=[f"Fold {i+1}" for i in range(5)], y=cv_scores,
                marker_color=["#667eea" if s == cv_scores.max() else "#a0aec0" for s in cv_scores],
                text=[f"{s*100:.1f}%" for s in cv_scores], textposition="outside")
    fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Media: {cv_scores.mean()*100:.1f}%")
    fig.update_layout(title="Scores por Fold", yaxis_title="Exactitud", yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📉 Curva de Aprendizaje")
    X_all = df[feature_names].values
    y_all = df["target"].values
    if escalar:
        X_all = StandardScaler().fit_transform(X_all)
    train_sizes, train_scores, val_scores = learning_curve(
        modelo.__class__(**modelo.get_params()), X_all, y_all,
        cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    fig = go.Figure()
    fig.add_scatter(x=train_sizes, y=train_scores.mean(axis=1),
                    name="Train", line=dict(color="#667eea"),
                    error_y=dict(type="data", array=train_scores.std(axis=1), visible=True))
    fig.add_scatter(x=train_sizes, y=val_scores.mean(axis=1),
                    name="Validación", line=dict(color="#f6ad55"),
                    error_y=dict(type="data", array=val_scores.std(axis=1), visible=True))
    fig.update_layout(title="Curva de Aprendizaje",
                      xaxis_title="Tamaño del conjunto de entrenamiento",
                      yaxis_title="Exactitud", yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="info-box">
    <b>¿Cómo leer la curva?</b> Si la brecha entre Train y Validación es grande → <b>Overfitting</b>.
    Si ambas son bajas → <b>Underfitting</b>. El modelo ideal tiene ambas curvas altas y cercanas.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — ANÁLISIS VISUAL
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## 🔍 Análisis Visual del Espacio de Características")

    feat_x = st.selectbox("Eje X", feature_names, index=0)
    feat_y = st.selectbox("Eje Y", feature_names, index=2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌿 Datos Reales")
        fig = px.scatter(df, x=feat_x, y=feat_y, color="especie",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Distribución real", hover_data=feature_names)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🤖 Predicciones del Modelo (Test Set)")
        idx_x = list(feature_names).index(feat_x)
        idx_y = list(feature_names).index(feat_y)
        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df["real"] = [target_names[t] for t in y_test]
        test_df["pred"] = [target_names[t] for t in y_pred]
        test_df["correcto"] = test_df["real"] == test_df["pred"]
        fig = px.scatter(test_df, x=feat_x, y=feat_y, color="pred",
                         symbol="correcto",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Predicciones (✓=correcto, ×=error)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌐 Scatter Matrix (Pairplot)")
    fig = px.scatter_matrix(df, dimensions=feature_names, color="especie",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            title="Relaciones entre todas las características")
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — INTERPRETABILIDAD
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## 🌳 Interpretabilidad del Modelo")

    if algoritmo == "Árbol de Decisión":
        st.markdown("#### 🌳 Visualización del Árbol de Decisión")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(modelo, feature_names=feature_names, class_names=target_names,
                  filled=True, rounded=True, ax=ax, fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if hasattr(modelo, "feature_importances_"):
        st.markdown("#### 📊 Importancia de Características")
        imp_df = pd.DataFrame({
            "Característica": feature_names,
            "Importancia": modelo.feature_importances_
        }).sort_values("Importancia", ascending=True)
        fig = px.bar(imp_df, x="Importancia", y="Característica", orientation="h",
                     color="Importancia", color_continuous_scale="Viridis",
                     title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    elif hasattr(modelo, "coef_"):
        st.markdown("#### 📊 Coeficientes del Modelo")
        coef_df = pd.DataFrame(modelo.coef_, columns=feature_names,
                                index=target_names)
        fig = px.imshow(coef_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Coeficientes por clase y característica")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"ℹ️ El algoritmo **{algoritmo}** no expone directamente la importancia de características. "
                "Prueba con Árbol de Decisión o Random Forest para ver esta sección.")

    st.markdown("#### 🔬 Comparativa de Algoritmos")
    if st.button("⚡ Comparar todos los modelos"):
        X_all = df[feature_names].values
        y_all = df["target"].values
        if escalar:
            scaler_tmp = StandardScaler()
            X_all = scaler_tmp.fit_transform(X_all)
        comparacion = {}
        todos = {
            "KNN": KNeighborsClassifier(),
            "Árbol Decisión": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB(),
            "Reg. Logística": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42),
        }
        with st.spinner("Comparando modelos..."):
            for nombre, m in todos.items():
                scores = cross_val_score(m, X_all, y_all, cv=5)
                comparacion[nombre] = {"Media": scores.mean(), "Std": scores.std()}
        comp_df = pd.DataFrame(comparacion).T.reset_index()
        comp_df.columns = ["Modelo", "Media CV", "Std CV"]
        comp_df = comp_df.sort_values("Media CV", ascending=True)
        fig = px.bar(comp_df, x="Media CV", y="Modelo", orientation="h",
                     error_x="Std CV", color="Media CV",
                     color_continuous_scale="Teal", title="Comparativa de modelos (CV 5-fold)")
        fig.update_xaxes(range=[0.8, 1.05])
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — PREDECIR
# ══════════════════════════════════════════════
with tab5:
    st.markdown("## 🧪 Hacer una Predicción")
    st.markdown("Ajusta los valores de las características y el modelo te dirá qué especie de IRIS es.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📐 Ingresar Características")
        stats = df[feature_names].describe()
        inputs = {}
        for feat in feature_names:
            mn = float(stats[feat]["min"])
            mx = float(stats[feat]["max"])
            mean = float(stats[feat]["mean"])
            inputs[feat] = st.slider(
                f"{feat} (cm)",
                min_value=round(mn, 1),
                max_value=round(mx, 1),
                value=round(mean, 1),
                step=0.1
            )

        st.markdown("---")
        if st.button("🔮 ¡Predecir!", type="primary", use_container_width=True):
            x_new = np.array([[inputs[f] for f in feature_names]])
            if escalar and scaler is not None:
                x_new = scaler.transform(x_new)
            pred = modelo.predict(x_new)[0]
            nombre_especie = target_names[pred]
            
            if hasattr(modelo, "predict_proba"):
                proba = modelo.predict_proba(x_new)[0]
            else:
                proba = None

            st.session_state["ultima_pred"] = (nombre_especie, proba)

    with col2:
        st.markdown("### 🌸 Resultado")
        if "ultima_pred" in st.session_state:
            nombre_especie, proba = st.session_state["ultima_pred"]
            emojis = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌷"}
            emoji = emojis.get(nombre_especie, "🌸")
            st.markdown(f"""
            <div class="prediction-box">
                {emoji} La flor es:<br>
                <span style="font-size:2rem; color:#764ba2">Iris {nombre_especie.capitalize()}</span>
            </div>
            """, unsafe_allow_html=True)

            if proba is not None:
                st.markdown("#### 📊 Probabilidades por Clase")
                prob_df = pd.DataFrame({
                    "Especie": target_names,
                    "Probabilidad": proba
                })
                fig = px.bar(prob_df, x="Especie", y="Probabilidad",
                             color="Especie",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             text=[f"{p*100:.1f}%" for p in proba])
                fig.update_yaxes(range=[0, 1.1])
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Ajusta los sliders y presiona **¡Predecir!** para ver el resultado.")
            st.markdown("""
            <div class="info-box">
            <b>¿Cómo funciona?</b><br>
            El modelo fue entrenado con datos históricos. Cuando ingresas nuevas medidas,
            encuentra el patrón más cercano aprendido durante el entrenamiento y predice la especie.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧬 Exploración Masiva: Prueba múltiples muestras")
    n_random = st.slider("¿Cuántas muestras aleatorias probar?", 5, 30, 10)
    if st.button("🎲 Generar predicciones aleatorias"):
        idx = np.random.choice(len(df), n_random, replace=False)
        X_sample = df.iloc[idx][feature_names].values
        y_true_sample = df.iloc[idx]["target"].values
        X_pred_sample = scaler.transform(X_sample) if escalar and scaler else X_sample
        y_pred_sample = modelo.predict(X_pred_sample)
        
        res_df = pd.DataFrame(X_sample, columns=feature_names)
        res_df["Real"] = [target_names[t] for t in y_true_sample]
        res_df["Predicción"] = [target_names[t] for t in y_pred_sample]
        res_df["✓"] = res_df["Real"] == res_df["Predicción"]
        res_df["✓"] = res_df["✓"].map({True: "✅", False: "❌"})
        st.dataframe(res_df, use_container_width=True)

        acc_muestra = (y_true_sample == y_pred_sample).mean()
        st.success(f"Exactitud en esta muestra: **{acc_muestra*100:.1f}%** ({int(acc_muestra*n_random)}/{n_random} correctas)")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#aaa; font-size:0.85rem;'>
🌸 Clasificador IRIS Pedagógico · Construido con Streamlit & scikit-learn
</p>
""", unsafe_allow_html=True)
