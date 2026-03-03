
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import io
import os

PALETTE = {
    "bg": "#0F1117", "card": "#1A1D2E", "accent": "#6C63FF",
    "danger": "#FF4C6A", "ok": "#00D68F", "text": "#E8E9F3", "muted": "#7B7E9A",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["card"], "axes.facecolor": PALETTE["card"],
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
    "text.color": PALETTE["text"], "grid.color": "#2A2D3E",
    "grid.linestyle": "--", "grid.alpha": 0.6, "font.family": "monospace",
})

st.set_page_config(page_title="Predicción de Deserción · Educación Superior", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background-color: #0F1117; }
  section[data-testid="stSidebar"] { background-color: #1A1D2E; border-right: 1px solid #2A2D3E; }
  div[data-testid="metric-container"] { background: #1A1D2E; border: 1px solid #2A2D3E; border-radius: 12px; padding: 18px 22px; }
  div[data-testid="metric-container"] label { color: #7B7E9A !important; font-size: 0.75rem; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; color: #E8E9F3 !important; }
  .section-header { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6C63FF; border-bottom: 1px solid #2A2D3E; padding-bottom: 6px; margin: 28px 0 10px; }
  .stDataFrame { background: #1A1D2E; border-radius: 10px; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0F1117; }
  ::-webkit-scrollbar-thumb { background: #2A2D3E; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding:20px 0 4px'>
  <span style='font-size:1.9rem; font-weight:900; color:#E8E9F3; letter-spacing:-1px'>Sistema de Predicción de Deserción en Estudiantes de Educación Superior</span><br>
  <span style='font-size:0.92rem; color:#7B7E9A; margin-top:6px; display:block'>
    Este sistema utiliza un modelo de clasificación <strong style='color:#6C63FF'>XGBoost</strong> entrenado previamente para predecir la probabilidad de deserción de cada estudiante, permitiendo identificar casos de alto riesgo de forma temprana y apoyar la toma de decisiones institucionales.
  </span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📋 ¿Cómo usar este sistema? — Instrucciones y plantilla de cargue", expanded=True):
    col_inst, col_dl = st.columns([3, 1])
    with col_inst:
        st.markdown("""
**Para utilizar el sistema siga estos pasos:**
1. **Descargue el archivo de referencia** (botón a la derecha) — contiene la descripción de cada variable y un ejemplo del formato requerido.
2. **Prepare su archivo Excel** con las columnas exactamente como se muestran en la hoja *Variables*, respetando los tipos de dato y códigos numéricos.
3. **No incluya la columna Target** — el sistema predice automáticamente si el estudiante desertará o no.
4. **Cargue su archivo** usando el botón de carga más abajo.
5. El modelo analizará cada estudiante y mostrará su **probabilidad de deserción** y **clasificación** (Deserta / No Deserta).
        """)
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        if os.path.exists("Variables_y_ejemplo.xlsx"):
            with open("Variables_y_ejemplo.xlsx", "rb") as f:
                st.download_button(label="⬇️ Descargar plantilla de variables y ejemplo", data=f,
                    file_name="Variables_y_ejemplo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
        else:
            st.warning("Archivo de plantilla no encontrado.")

st.markdown("---")

@st.cache_resource
def cargar_modelo():
    archivos = ["modelo_desercion.pkl","columnas_modelo.json","metricas_train.json","roc_train.json","feature_importance.csv"]
    faltantes = [f for f in archivos if not os.path.exists(f)]
    if faltantes:
        return None, None, None, None, None, faltantes
    modelo = joblib.load("modelo_desercion.pkl")
    with open("columnas_modelo.json") as f: columnas = json.load(f)
    with open("metricas_train.json") as f: metricas = json.load(f)
    with open("roc_train.json") as f: roc = json.load(f)
    imp_df = pd.read_csv("feature_importance.csv", index_col=0)
    return modelo, columnas, metricas, roc, imp_df, []

modelo, columnas_modelo, metricas_train, roc_data, imp_df, faltantes = cargar_modelo()
if faltantes:
    st.error(f"❌ Faltan archivos del modelo: **{', '.join(faltantes)}**")
    st.info("Ejecuta primero el script `guardar_modelo.py` en tu Jupyter y copia los 5 archivos generados a esta misma carpeta.")
    st.stop()

uploaded_file = st.file_uploader("📂 Cargar archivo Excel con datos de estudiantes (sin columna Target)", type=["xlsx"])
if uploaded_file is None:
    st.info("⬆️ Cargue su archivo Excel para comenzar el análisis.")
    st.stop()

df_raw = pd.read_excel(uploaded_file)
if "Target" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Target"])

try:
    df_enc = pd.get_dummies(df_raw, drop_first=True)
    for col in columnas_modelo:
        if col not in df_enc.columns: df_enc[col] = 0
    df_enc = df_enc[columnas_modelo]
except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}"); st.stop()

df_result = df_raw.copy()
df_result["P(Deserción)"] = modelo.predict_proba(df_enc)[:, 1]

# ── Mapeos de etiquetas legibles
MAP_CURSOS = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising & Marketing Mgmt",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)",
}
if "Gender" in df_result.columns:
    df_result["Genero_lbl"] = df_result["Gender"].map({0: "Femenino", 1: "Masculino"})
if "Scholarship holder" in df_result.columns:
    df_result["Beca_lbl"] = df_result["Scholarship holder"].map({0: "No", 1: "Sí"})
if "Course" in df_result.columns:
    df_result["Course_lbl"] = df_result["Course"].map(MAP_CURSOS).fillna(df_result["Course"].astype(str))

with st.sidebar:
    st.markdown("### 🎛️ Filtros")
    curso = "Todos"
    if "Course_lbl" in df_result.columns:
        curso = st.selectbox("Programa", ["Todos"] + sorted(df_result["Course_lbl"].dropna().unique().tolist()))
    genero = "Todos"
    if "Genero_lbl" in df_result.columns:
        genero = st.selectbox("Género", ["Todos"] + sorted(df_result["Genero_lbl"].dropna().unique().tolist()))
    beca = "Todos"
    if "Beca_lbl" in df_result.columns:
        beca = st.selectbox("Beca", ["Todos"] + sorted(df_result["Beca_lbl"].dropna().unique().tolist()))
    edad_range = None
    if "Age at enrollment" in df_result.columns:
        edad_min = int(df_result["Age at enrollment"].min())
        edad_max = int(df_result["Age at enrollment"].max())
        edad_range = st.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))
    st.markdown("---")
    umbral = st.slider("Umbral de clasificación", 0.30, 0.80, 0.50, 0.01,
                       help="Probabilidad mínima para clasificar como 'Deserta' (por defecto: 0.50)")

df_filt = df_result.copy()
if curso  != "Todos" and "Course_lbl" in df_filt.columns: df_filt = df_filt[df_filt["Course_lbl"] == curso]
if genero != "Todos" and "Genero_lbl" in df_filt.columns: df_filt = df_filt[df_filt["Genero_lbl"] == genero]
if beca   != "Todos" and "Beca_lbl"   in df_filt.columns: df_filt = df_filt[df_filt["Beca_lbl"]   == beca]
if edad_range and "Age at enrollment" in df_filt.columns:
    df_filt = df_filt[(df_filt["Age at enrollment"] >= edad_range[0]) & (df_filt["Age at enrollment"] <= edad_range[1])]
df_filt["Clasificación"] = df_filt["P(Deserción)"].apply(lambda p: "Deserta" if p >= umbral else "No Deserta")

tab1, tab2, tab3 = st.tabs(["📊 Análisis del Riesgo", "🤖 Métricas del Modelo", "🚨 Estudiantes en Riesgo"])

with tab1:
    total = len(df_filt)
    deserta = (df_filt["Clasificación"] == "Deserta").sum()
    no_des  = (df_filt["Clasificación"] == "No Deserta").sum()
    avg_p   = df_filt["P(Deserción)"].mean() if total > 0 else 0

    st.markdown('<div class="section-header">Indicadores Estratégicos</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Estudiantes", total)
    c2.metric("🔴 Deserta",     f"{deserta}", f"{deserta/total:.1%}" if total > 0 else "")
    c3.metric("🟢 No Deserta",  f"{no_des}",  f"{no_des/total:.1%}"  if total > 0 else "")
    c4.metric("📈 Prob. Promedio", f"{avg_p:.1%}")

    st.markdown('<div class="section-header">Distribución de la Clasificación</div>', unsafe_allow_html=True)
    st.caption("Muestra cuántos estudiantes han sido clasificados como en riesgo de desertar versus los que se espera continúen sus estudios. El umbral ajustable en el panel lateral determina el punto de corte.")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(["No Deserta", "Deserta"], [no_des, deserta], color=[PALETTE["ok"], PALETTE["danger"]], width=0.5, zorder=3)
        ax.set_ylabel("Estudiantes"); ax.grid(axis="y"); ax.set_title("Clasificación de Estudiantes")
        for bar, val in zip(bars, [no_des, deserta]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie([no_des, deserta], labels=["No Deserta", "Deserta"], colors=[PALETTE["ok"], PALETTE["danger"]],
                autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.55, edgecolor="#0F1117", linewidth=2))
        ax2.set_title("Proporción Deserta / No Deserta")
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True)

    if "Course_lbl" in df_filt.columns:
        st.markdown('<div class="section-header">Deserción por Programa</div>', unsafe_allow_html=True)
        st.caption("Comparativo por programa académico del número de estudiantes predichos como desertores versus los que continuarán. Permite identificar qué programas concentran mayor riesgo institucional.")
        prog_counts = (df_filt.groupby("Course_lbl")["Clasificación"].value_counts().unstack(fill_value=0)
                       .reindex(columns=["Deserta", "No Deserta"], fill_value=0).sort_values("Deserta", ascending=True))
        fig3, ax3 = plt.subplots(figsize=(9, max(3.5, len(prog_counts)*0.5)))
        w = 0.38; y = range(len(prog_counts))
        ax3.barh([i - w/2 for i in y], prog_counts["Deserta"],    height=w, color=PALETTE["danger"], label="Deserta",    zorder=3)
        ax3.barh([i + w/2 for i in y], prog_counts["No Deserta"], height=w, color=PALETTE["ok"],     label="No Deserta", zorder=3)
        ax3.set_yticks(list(y)); ax3.set_yticklabels(prog_counts.index, fontsize=9)
        ax3.set_xlabel("Número de Estudiantes"); ax3.set_title("Deserta vs No Deserta por Programa")
        ax3.legend(fontsize=9); ax3.grid(axis="x")
        fig3.tight_layout(); st.pyplot(fig3, use_container_width=True)

    if "Genero_lbl" in df_filt.columns or "Beca_lbl" in df_filt.columns:
        st.markdown('<div class="section-header">Riesgo por Variables Demográficas</div>', unsafe_allow_html=True)
        st.caption("Probabilidad promedio de deserción según género y condición de beca. Una barra que supere la línea punteada indica que ese grupo, en promedio, se encuentra en zona de riesgo.")
        col_g, col_b2 = st.columns(2)
        with col_g:
            if "Genero_lbl" in df_filt.columns:
                gen_df = df_filt.groupby("Genero_lbl")["P(Deserción)"].mean()
                fig4, ax4 = plt.subplots(figsize=(4, 3))
                ax4.bar(gen_df.index, gen_df.values, color=[PALETTE["accent"], PALETTE["ok"]], width=0.45, zorder=3)
                ax4.axhline(umbral, color=PALETTE["danger"], lw=1.2, ls="--", label=f"Umbral ({umbral:.0%})")
                ax4.set_ylabel("Prob. Promedio"); ax4.set_title("Por Género"); ax4.legend(fontsize=8); ax4.grid(axis="y")
                fig4.tight_layout(); st.pyplot(fig4, use_container_width=True)
        with col_b2:
            if "Beca_lbl" in df_filt.columns:
                bec_df = df_filt.groupby("Beca_lbl")["P(Deserción)"].mean()
                fig5, ax5 = plt.subplots(figsize=(4, 3))
                ax5.bar(bec_df.index, bec_df.values, color=[PALETTE["muted"], PALETTE["ok"]], width=0.45, zorder=3)
                ax5.axhline(umbral, color=PALETTE["danger"], lw=1.2, ls="--", label=f"Umbral ({umbral:.0%})")
                ax5.set_ylabel("Prob. Promedio"); ax5.set_title("Con / Sin Beca"); ax5.legend(fontsize=8); ax5.grid(axis="y")
                fig5.tight_layout(); st.pyplot(fig5, use_container_width=True)

    if "Age at enrollment" in df_filt.columns:
        st.markdown('<div class="section-header">Análisis por Edad</div>', unsafe_allow_html=True)
        st.caption("A la izquierda, la distribución de edades de los estudiantes cargados. A la derecha, la probabilidad promedio de deserción por rango etario. Las barras en rojo superan el umbral de riesgo.")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig6, ax6 = plt.subplots(figsize=(5, 3.5))
            ax6.hist(df_filt["Age at enrollment"], bins=20, color=PALETTE["accent"], edgecolor="#0F1117", zorder=3)
            ax6.set_xlabel("Edad"); ax6.set_ylabel("Frecuencia"); ax6.set_title("Distribución de Edades"); ax6.grid(axis="y")
            fig6.tight_layout(); st.pyplot(fig6, use_container_width=True)
        with col_h2:
            bins_e = pd.cut(df_filt["Age at enrollment"], bins=[15,20,25,30,40,100], labels=["15-20","20-25","25-30","30-40","40+"])
            age_df = df_filt.groupby(bins_e)["P(Deserción)"].mean().dropna()
            fig7, ax7 = plt.subplots(figsize=(5, 3.5))
            ax7.bar(age_df.index.astype(str), age_df.values,
                    color=[PALETTE["danger"] if v >= umbral else PALETTE["ok"] for v in age_df.values], width=0.55, zorder=3)
            ax7.axhline(umbral, color=PALETTE["danger"], lw=1.2, ls="--", label=f"Umbral ({umbral:.0%})")
            ax7.set_xlabel("Rango de Edad"); ax7.set_ylabel("Prob. Promedio"); ax7.set_title("Riesgo Promedio por Rango de Edad")
            ax7.legend(fontsize=8); ax7.grid(axis="y")
            fig7.tight_layout(); st.pyplot(fig7, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Desempeño del Modelo (Train)</div>', unsafe_allow_html=True)
    st.caption("Métricas del conjunto de entrenamiento del modelo base. Los datos cargados al dashboard son nuevos datos de producción sobre los que el modelo realiza predicciones.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metricas_train['Accuracy']:.3f}")
    c2.metric("Precision", f"{metricas_train['Precision']:.3f}")
    c3.metric("Recall",    f"{metricas_train['Recall']:.3f}")
    c4.metric("F1-Score",  f"{metricas_train['F1']:.3f}")
    c5.metric("ROC AUC",   f"{metricas_train['ROC_AUC']:.3f}")
    col_roc, col_imp = st.columns(2)
    with col_roc:
        st.markdown('<div class="section-header">Curva ROC (Train)</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(roc_data["fpr"], roc_data["tpr"], color=PALETTE["accent"], lw=2, label=f"AUC = {metricas_train['ROC_AUC']:.3f}")
        ax_roc.plot([0,1],[0,1], color=PALETTE["muted"], lw=1, ls="--")
        ax_roc.set_xlabel("Tasa de Falsos Positivos"); ax_roc.set_ylabel("Tasa de Verdaderos Positivos")
        ax_roc.set_title("Curva ROC — Train"); ax_roc.legend(fontsize=9); ax_roc.grid()
        fig_roc.tight_layout(); st.pyplot(fig_roc, use_container_width=True)
    with col_imp:
        st.markdown('<div class="section-header">Top 15 Variables Influyentes</div>', unsafe_allow_html=True)
        top15 = imp_df.sort_values("importance", ascending=False).head(15)
        norm = top15["importance"].max()
        bar_col = [PALETTE["accent"] if v >= norm*0.5 else PALETTE["muted"] for v in top15["importance"]]
        fig_imp, ax_imp = plt.subplots(figsize=(5, 4.5))
        ax_imp.barh(top15.index[::-1], top15["importance"][::-1], color=bar_col[::-1], height=0.65, zorder=3)
        ax_imp.set_xlabel("Importancia (gain)"); ax_imp.set_title("Feature Importance — XGBoost"); ax_imp.grid(axis="x")
        fig_imp.tight_layout(); st.pyplot(fig_imp, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Estudiantes Clasificados como Deserta</div>', unsafe_allow_html=True)
    cols_base = [c for c in ["Course_lbl", "Genero_lbl", "Beca_lbl", "Age at enrollment"] if c in df_filt.columns]
    cols_show = cols_base + ["P(Deserción)", "Clasificación"]
    rename_map = {"Course_lbl": "Programa", "Genero_lbl": "Género", "Beca_lbl": "Beca", "Age at enrollment": "Edad"}
    criticos = (df_filt[df_filt["Clasificación"] == "Deserta"]
                .sort_values("P(Deserción)", ascending=False)[cols_show].rename(columns=rename_map))
    st.info(f"🔴 Se encontraron **{len(criticos)}** estudiantes clasificados como **Deserta** (umbral: {umbral:.0%})")
    st.dataframe(criticos.style.background_gradient(subset=["P(Deserción)"], cmap="Reds").format({"P(Deserción)": "{:.1%}"}),
                 use_container_width=True, height=420)
    df_export_todos = df_filt[cols_show].rename(columns=rename_map).sort_values("P(Deserción)", ascending=False)
    df_export_des   = df_export_todos[df_export_todos["Clasificación"] == "Deserta"]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export_todos.to_excel(writer, sheet_name="Todos_los_estudiantes", index=False)
        df_export_des.to_excel(writer, sheet_name="Deserta", index=False)
    buffer.seek(0)
    st.download_button("⬇️ Descargar resultados (Excel con 2 hojas)", buffer, "predicciones_desercion.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
