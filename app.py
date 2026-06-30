# -*- coding: utf-8 -*-
"""
Dashboard web — Predicción de Mortalidad en UCI (WiDS Datathon)
Versión web del notebook mortality_prediction.ipynb.
Modelo final: XGBoost (sin cambios respecto al notebook).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ======================================================================
# DESCRIPCIONES DE VARIABLES (tooltips informativos)
# ======================================================================
INFO_VARS = {
    "hospital_death": "`hospital_death` — Variable objetivo: 1 = el paciente falleció en el hospital, 0 = sobrevivió.",
    "age": "`age` — Edad del paciente en años.",
    "bmi": "`bmi` — Índice de masa corporal (peso kg / altura² m²).",
    "gender": "`gender` — Sexo del paciente (M/F).",
    "ethnicity": "`ethnicity` — Grupo étnico declarado del paciente.",
    "icu_type": "`icu_type` — Tipo de unidad de cuidados intensivos de admisión (MICU, SICU, CCU-CTICU, etc.).",
    "height": "`height` — Estatura del paciente en cm.",
    "weight": "`weight` — Peso del paciente en kg.",
    "pre_icu_los_days": "`pre_icu_los_days` — Días entre el ingreso al hospital y el ingreso a la UCI.",
    "heart_rate_apache": "`heart_rate_apache` — Frecuencia cardíaca de las primeras 24 h que produce el puntaje APACHE III más alto (valor más alterado).",
    "map_apache": "`map_apache` — Presión arterial media (MAP) de las primeras 24 h usada en el puntaje APACHE III.",
    "temp_apache": "`temp_apache` — Temperatura corporal de las primeras 24 h usada en el puntaje APACHE III.",
    "resprate_apache": "`resprate_apache` — Frecuencia respiratoria de las primeras 24 h usada en el puntaje APACHE III.",
    "sodium_apache": "`sodium_apache` — Sodio sérico de las primeras 24 h (APACHE III).",
    "creatinine_apache": "`creatinine_apache` — Creatinina sérica (función renal), usada en APACHE III.",
    "bun_apache": "`bun_apache` — Nitrógeno ureico en sangre (BUN); marcador de función renal (APACHE III).",
    "bilirubin_apache": "`bilirubin_apache` — Bilirrubina sérica (función hepática), usada en APACHE III.",
    "hematocrit_apache": "`hematocrit_apache` — Hematocrito (% de glóbulos rojos), usado en APACHE III.",
    "fio2_apache": "`fio2_apache` — Fracción inspirada de oxígeno (FiO₂), usada en APACHE III.",
    "gcs_eyes_apache": "`gcs_eyes_apache` — Apertura ocular de la escala de Glasgow (1–4); a menor valor, peor estado neurológico.",
    "gcs_motor_apache": "`gcs_motor_apache` — Componente motor de Glasgow (1–6); a menor valor, peor respuesta.",
    "gcs_verbal_apache": "`gcs_verbal_apache` — Componente verbal de Glasgow (1–5); a menor valor, peor respuesta.",
    "d1_heartrate_max": "`d1_heartrate_max` — Frecuencia cardíaca más alta en las primeras 24 h en UCI.",
    "d1_heartrate_min": "`d1_heartrate_min` — Frecuencia cardíaca más baja en las primeras 24 h en UCI.",
    "h1_sysbp_max": "`h1_sysbp_max` — Presión arterial sistólica máxima durante la primera hora en UCI.",
    "h1_diasbp_noninvasive_min": "`h1_diasbp_noninvasive_min` — Presión arterial diastólica mínima (no invasiva) en la primera hora.",
    "d1_resprate_max": "`d1_resprate_max` — Frecuencia respiratoria máxima en las primeras 24 h.",
    "d1_resprate_min": "`d1_resprate_min` — Frecuencia respiratoria mínima en las primeras 24 h.",
    "h1_resprate_max": "`h1_resprate_max` — Frecuencia respiratoria máxima en la primera hora.",
    "h1_resprate_min": "`h1_resprate_min` — Frecuencia respiratoria mínima en la primera hora.",
    "d1_spo2_min": "`d1_spo2_min` — Saturación de oxígeno (SpO₂) más baja en las primeras 24 h.",
    "d1_temp_min": "`d1_temp_min` — Temperatura corporal más baja en las primeras 24 h.",
    "d1_temp_max": "`d1_temp_max` — Temperatura corporal más alta en las primeras 24 h.",
    "d1_glucose_min": "`d1_glucose_min` — Glucosa en sangre más baja en las primeras 24 h.",
    "ventilated_apache": "`ventilated_apache` — Indica si recibió ventilación mecánica invasiva (1 = sí).",
    "intubated_apache": "`intubated_apache` — Indica si el paciente fue intubado (1 = sí).",
    "apache_post_operative": "`apache_post_operative` — Indica si ingresó a UCI tras una cirugía (1 = postoperatorio).",
    "diabetes_mellitus": "`diabetes_mellitus` — Antecedente de diabetes mellitus (1 = sí).",
    "apache_2_diagnosis": "`apache_2_diagnosis` — Código de diagnóstico de admisión a UCI según APACHE II.",
    "apache_2_bodysystem": "`apache_2_bodysystem` — Sistema corporal afectado según la clasificación APACHE II.",
    "apache_3j_bodysystem": "`apache_3j_bodysystem` — Sistema corporal afectado según la clasificación APACHE III.",
}

# ======================================================================
# CONFIGURACIÓN Y ESTILO
# ======================================================================
st.set_page_config(page_title="Mortalidad UCI — Dashboard",
                   page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

ART = Path(__file__).parent / "artifacts"

# Paleta (mismo espíritu visual que el dashboard de referencia)
C_SURV = "#3b8ea5"   # azul/teal  -> sobrevivió
C_DIED = "#e0564f"   # rojo/coral -> falleció
TEMPLATE = "plotly_dark"

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
section[data-testid="stSidebar"] {background-color: #0c1016;}
/* Tarjeta KPI */
.kpi-label {font-size:.95rem; color:#9aa4b2; margin-bottom:.1rem;}
.kpi-value {font-size:2.6rem; font-weight:700; line-height:1.1; color:#fafafa;}
/* Cajas de resultado del Panel B */
.res-ok   {background:#143d2b; border:1px solid #1f6b48; border-radius:10px;
           padding:18px 22px; font-size:1.6rem; font-weight:700; color:#5ee2a0;}
.res-bad  {background:#451c1c; border:1px solid #8a2f2f; border-radius:10px;
           padding:18px 22px; font-size:1.6rem; font-weight:700; color:#ff8a82;}
.res-info {background:#10263a; border:1px solid #1f4a6b; border-radius:10px;
           padding:18px 22px; color:#cfe2f3; font-size:.95rem; line-height:1.5;}
.prob-big {font-size:2.4rem; font-weight:700; color:#fafafa;}
.sidebar-title {font-size:1.35rem; font-weight:700; color:#fafafa; margin-bottom:0;}
.sidebar-sub {font-size:.8rem; color:#8a93a3;}
h2, h3 {color:#fafafa;}
hr {border-color:#262d3a;}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# CARGA DE ARTEFACTOS
# ======================================================================
@st.cache_data(show_spinner=False)
def load_meta():
    return json.loads((ART / "metadata.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_panelA():
    df = pd.read_parquet(ART / "panelA.parquet")
    for c in df.columns:                       # Int64 -> int normal para graficar
        if str(df[c].dtype) == "Int64":
            df[c] = df[c].astype("float")
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    booster = xgb.Booster()
    booster.load_model(str(ART / "modelo_final.json"))
    return booster

META = load_meta()
BOOSTER = load_model()
FEATURES = META["variables_finales"]          # orden exacto que espera el modelo

GENDER_MAP = {"M": "Masculino", "F": "Femenino"}


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏥 Mortalidad UCI<br>Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Predicción de mortalidad hospitalaria en UCI</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Fuente de datos")
    st.caption("Usa el dataset incluido o sube un CSV limpio con el mismo esquema (opcional).")
    up = st.file_uploader("Subir CSV (opcional)", type=["csv"], label_visibility="collapsed")

    df = load_panelA()
    if up is not None:
        try:
            tmp = pd.read_csv(up)
            if {"hospital_death", "age"}.issubset(tmp.columns):
                df = tmp
                st.success("CSV cargado correctamente.")
            else:
                st.warning("El CSV no tiene el esquema esperado. Se usa el dataset incluido.")
        except Exception:
            st.warning("No se pudo leer el CSV. Se usa el dataset incluido.")

    st.divider()
    st.subheader("Filtros — Panel A")

    a_min, a_max = int(df["age"].min()), int(df["age"].max())
    rango_edad = st.slider("Rango de edad", a_min, a_max, (a_min, a_max))

    sexo_op = ["Todos"] + [GENDER_MAP.get(g, g) for g in sorted(df["gender"].dropna().unique())]
    sexo = st.selectbox("Sexo", sexo_op)

    icu_op = ["Todos"] + sorted(df["icu_type"].dropna().unique().tolist())
    icu = st.selectbox("Tipo de UCI", icu_op)

    desen = st.selectbox("Desenlace", ["Todos", "Sobrevivió", "Falleció"])

# ----- aplicar filtros -----
f = df[(df["age"] >= rango_edad[0]) & (df["age"] <= rango_edad[1])].copy()
if sexo != "Todos":
    inv = {v: k for k, v in GENDER_MAP.items()}
    f = f[f["gender"] == inv.get(sexo, sexo)]
if icu != "Todos":
    f = f[f["icu_type"] == icu]
if desen == "Sobrevivió":
    f = f[f["hospital_death"] == 0]
elif desen == "Falleció":
    f = f[f["hospital_death"] == 1]

f["Desenlace"] = np.where(f["hospital_death"] == 1, "Falleció", "Sobrevivió")
ORDER = ["Sobrevivió", "Falleció"]
CMAP = {"Sobrevivió": C_SURV, "Falleció": C_DIED}


# ======================================================================
# PESTAÑAS
# ======================================================================
tab_a, tab_b = st.tabs(["📊 Panel A — Análisis de Datos",
                        "🤖 Panel B — Análisis Predictivo"])

# ----------------------------------------------------------------------
# PANEL A
# ----------------------------------------------------------------------
with tab_a:
    st.markdown("## 📊 Panel A — Análisis de Datos")
    st.caption("Explora el dataset WiDS — Mortalidad en UCI de forma interactiva.")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown('<p class="kpi-label">Registros filtrados</p>'
                    f'<p class="kpi-value">{len(f):,} / {len(df):,}</p>',
                    unsafe_allow_html=True)
    with k2:
        tasa = (f["hospital_death"].mean() * 100) if len(f) else 0
        st.markdown('<p class="kpi-label">% mortalidad UCI</p>'
                    f'<p class="kpi-value">{tasa:.1f}%</p>', unsafe_allow_html=True)
    with k3:
        edad_prom = f["age"].mean() if len(f) else 0
        st.markdown('<p class="kpi-label">Edad promedio</p>'
                    f'<p class="kpi-value">{edad_prom:.1f} años</p>',
                    unsafe_allow_html=True)

    st.divider()

    if len(f) == 0:
        st.info("No hay registros con los filtros seleccionados.")
    else:
        r1c1, r1c2 = st.columns(2)
        # --- Histograma de edad por desenlace ---
        with r1c1:
            st.markdown("#### Distribución de edad por desenlace")
            fig = px.histogram(f, x="age", color="Desenlace", nbins=30,
                               barmode="overlay", opacity=0.75,
                               category_orders={"Desenlace": ORDER},
                               color_discrete_map=CMAP,
                               labels={"age": "Edad", "count": "Casos"})
            fig.update_layout(template=TEMPLATE, height=360,
                              legend_title_text="Desenlace",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # --- Scatter BMI vs FC máxima día 1 ---
        with r1c2:
            st.markdown("#### IMC vs. frecuencia cardíaca máxima (día 1)")
            sc = f.dropna(subset=["bmi", "d1_heartrate_max"])
            fig = px.scatter(sc, x="bmi", y="d1_heartrate_max", color="Desenlace",
                             category_orders={"Desenlace": ORDER},
                             color_discrete_map=CMAP, opacity=0.55,
                             labels={"bmi": "IMC", "d1_heartrate_max": "Frec. cardíaca máx."})
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(template=TEMPLATE, height=360,
                              legend_title_text="Desenlace",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        # --- Barras distribución del target ---
        with r2c1:
            st.markdown("#### Distribución de la variable objetivo (subconjunto filtrado)")
            vc = f["Desenlace"].value_counts().reindex(ORDER).fillna(0).reset_index()
            vc.columns = ["Desenlace", "Casos"]
            fig = px.bar(vc, x="Desenlace", y="Casos", color="Desenlace",
                         category_orders={"Desenlace": ORDER}, color_discrete_map=CMAP)
            fig.update_layout(template=TEMPLATE, height=360, showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # --- Mapa de calor de correlaciones ---
        with r2c2:
            st.markdown("#### Mapa de calor de correlaciones (variables numéricas)")
            num_cols = ["age", "bmi", "map_apache", "temp_apache", "resprate_apache",
                        "heart_rate_apache", "creatinine_apache", "bun_apache",
                        "pre_icu_los_days", "hospital_death"]
            num_cols = [c for c in num_cols if c in f.columns]
            corr = f[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            etiquetas = corr.columns.tolist()
            def _desc(c): return INFO_VARS.get(c, c).split("—", 1)[-1].strip()
            custom = [[f"<b>{yv}</b>: {_desc(yv)}<br><b>{xv}</b>: {_desc(xv)}" for xv in etiquetas] for yv in etiquetas]
            fig.update_traces(customdata=custom, hovertemplate="Correlación: %{z}<br>%{customdata}<extra></extra>")
            fig.update_layout(template=TEMPLATE, height=360,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Ver tabla de datos filtrados"):
            st.dataframe(
                f.drop(columns=["Desenlace"]).reset_index(drop=True),
                use_container_width=True, height=380,
                column_config={c: st.column_config.Column(help=INFO_VARS.get(c))
                               for c in f.drop(columns=["Desenlace"]).columns},
            )


# ----------------------------------------------------------------------
# PANEL B
# ----------------------------------------------------------------------
def construir_features(v: dict) -> pd.DataFrame:
    """Ensambla el vector en el orden exacto que espera el modelo."""
    row = {feat: 0.0 for feat in FEATURES}
    for k, val in v.items():
        if k in row:
            row[k] = float(val)
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)


with tab_b:
    st.markdown("## 🤖 Panel B — Análisis Predictivo")
    fm = META["final_metrics"]
    st.markdown(f"**Modelo activo:** XGBoost &nbsp;·&nbsp; "
                f"AUC = {fm['AUC']:.4f} &nbsp;|&nbsp; F1-Score = {fm['F1']:.4f} "
                f"&nbsp;|&nbsp; Recall = {fm['Recall']:.4f} (conjunto de prueba)")

    st.markdown("### Ingresa los datos clínicos del paciente")

    col1, col2, col3 = st.columns(3)

    # ---------- Columna 1: demografía y presión ----------
    with col1:
        age = st.slider("Edad (años)", 16, 89, 65, help=INFO_VARS["age"])
        bmi = st.slider("Índice de masa corporal (IMC)", 14.0, 68.0, 27.8, 0.1, help=INFO_VARS["bmi"])
        pre_icu = st.slider("Días previos al ingreso a UCI", 0.0, 84.0, 0.1, 0.1, help=INFO_VARS["pre_icu_los_days"])
        map_apache = st.slider("Presión arterial media — APACHE (mmHg)", 40, 200, 66, help=INFO_VARS["map_apache"])
        h1_sysbp_max = st.slider("PAS máx. primera hora (mmHg)", 75, 223, 131, help=INFO_VARS["h1_sysbp_max"])
        gcs_eyes = st.slider("GCS ocular (apertura)", 1, 4, 4, help=INFO_VARS["gcs_eyes_apache"])
        d1_hr_min = st.slider("Frec. cardíaca mín. día 1 (lpm)", 0, 175, 70, help=INFO_VARS["d1_heartrate_min"])
        h1_diasbp_nm = st.checkbox("PAD mín. 1ª hora no medida", help=INFO_VARS["h1_diasbp_noninvasive_min"])
        h1_diasbp = st.slider("PAD mín. primera hora — no invasiva (mmHg)",
                              22, 114, 62, disabled=h1_diasbp_nm, help=INFO_VARS["h1_diasbp_noninvasive_min"])

    # ---------- Columna 2: respiratorio, temperatura, glucosa ----------
    with col2:
        resprate_apache = st.slider("Frec. respiratoria — APACHE (rpm)", 4, 60, 28, help=INFO_VARS["resprate_apache"])
        d1_rr_max = st.slider("Frec. respiratoria máx. día 1 (rpm)", 14, 92, 26, help=INFO_VARS["d1_resprate_max"])
        d1_rr_min = st.slider("Frec. respiratoria mín. día 1 (rpm)", 0, 96, 13, help=INFO_VARS["d1_resprate_min"])
        h1_rr_max = st.slider("Frec. respiratoria máx. primera hora (rpm)", 10, 59, 21, help=INFO_VARS["h1_resprate_max"])
        h1_rr_min = st.slider("Frec. respiratoria mín. primera hora (rpm)", 0, 189, 16, help=INFO_VARS["h1_resprate_min"])
        d1_spo2 = st.slider("SpO₂ mínima día 1 (%)", 0, 100, 93, help=INFO_VARS["d1_spo2_min"])
        d1_temp_min = st.slider("Temperatura mín. día 1 (°C)", 31.9, 37.8, 36.4, 0.1, help=INFO_VARS["d1_temp_min"])
        d1_temp_max = st.slider("Temperatura máx. día 1 (°C)", 35.1, 39.9, 37.2, 0.1, help=INFO_VARS["d1_temp_max"])
        glu_nm = st.checkbox("Glucosa mín. día 1 no medida", help=INFO_VARS["d1_glucose_min"])
        d1_glu = st.slider("Glucosa mín. día 1 (mg/dl)", 33, 288, 105, disabled=glu_nm, help=INFO_VARS["d1_glucose_min"])

    # ---------- Columna 3: banderas clínicas y diagnósticos ----------
    with col3:
        si_no = {"No": 0, "Sí": 1}
        ventilated = si_no[st.selectbox("¿Ventilación invasiva?", ["No", "Sí"], help=INFO_VARS["ventilated_apache"])]
        intubated = si_no[st.selectbox("¿Paciente intubado?", ["No", "Sí"], help=INFO_VARS["intubated_apache"])]
        postop = si_no[st.selectbox("¿Postoperatorio (APACHE)?", ["No", "Sí"], help=INFO_VARS["apache_post_operative"])]
        diabetes = si_no[st.selectbox("¿Diabetes mellitus?", ["No", "Sí"], help=INFO_VARS["diabetes_mellitus"])]
        dx2 = st.selectbox("Diagnóstico APACHE-2", ["Otro", "Código 308", "Código 302"], help=INFO_VARS["apache_2_diagnosis"])
        bsys2 = st.selectbox("Sistema corporal APACHE-2", ["Otro", "Metabólico"], help=INFO_VARS["apache_2_bodysystem"])
        bsys3 = st.selectbox("Sistema corporal APACHE-3J", ["Otro", "Cardiovascular"], help=INFO_VARS["apache_3j_bodysystem"])

    predecir = st.button("🔎 Predecir", use_container_width=True, type="primary")

    if predecir:
        valores = {
            "age": age, "bmi": bmi, "pre_icu_los_days": pre_icu, "map_apache": map_apache,
            "h1_sysbp_max": h1_sysbp_max, "gcs_eyes_apache": gcs_eyes,
            "d1_heartrate_min": d1_hr_min,
            "h1_diasbp_noninvasive_min": (-99999 if h1_diasbp_nm else h1_diasbp),
            "resprate_apache": resprate_apache, "d1_resprate_max": d1_rr_max,
            "d1_resprate_min": d1_rr_min, "h1_resprate_max": h1_rr_max,
            "h1_resprate_min": h1_rr_min, "d1_spo2_min": d1_spo2,
            "d1_temp_min": d1_temp_min, "d1_temp_max": d1_temp_max,
            "d1_glucose_min": (-99999 if glu_nm else d1_glu),
            "ventilated_apache": ventilated, "intubated_apache": intubated,
            "apache_post_operative": postop, "diabetes_mellitus": diabetes,
            "apache_2_diagnosis_308": 1 if dx2 == "Código 308" else 0,
            "apache_2_diagnosis_302": 1 if dx2 == "Código 302" else 0,
            "apache_2_bodysystem_Metabolic": 1 if bsys2 == "Metabólico" else 0,
            "apache_3j_bodysystem_Cardiovascular": 1 if bsys3 == "Cardiovascular" else 0,
        }
        X = construir_features(valores)
        dmat = xgb.DMatrix(X, feature_names=FEATURES)
        p_muerte = float(BOOSTER.predict(dmat)[0])
        clase = int(p_muerte >= 0.5)

        st.divider()
        rc1, rc2 = st.columns([1.1, 1])
        with rc1:
            if clase == 0:
                st.markdown('<div class="res-ok">✅ Bajo riesgo de mortalidad hospitalaria</div>',
                            unsafe_allow_html=True)
                st.progress(1 - p_muerte)
                st.markdown('<p class="kpi-label">Probabilidad de supervivencia</p>'
                            f'<p class="prob-big">{(1-p_muerte)*100:.1f}%</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="res-bad">⚠️ Riesgo de mortalidad hospitalaria</div>',
                            unsafe_allow_html=True)
                st.progress(p_muerte)
                st.markdown('<p class="kpi-label">Probabilidad de fallecimiento</p>'
                            f'<p class="prob-big">{p_muerte*100:.1f}%</p>',
                            unsafe_allow_html=True)
        with rc2:
            st.markdown(
                '<div class="res-info"><b>¿Qué significa este resultado?</b><br>'
                'El modelo estima, a partir de los datos clínicos de las primeras 24 h en '
                'UCI, qué tan parecido es el patrón del paciente al de quienes fallecieron '
                'en el dataset de entrenamiento. <b>No es un diagnóstico</b>: es una ayuda '
                'a la decisión clínica para priorizar recursos y vigilancia.</div>',
                unsafe_allow_html=True)

    st.markdown("")
    with st.expander("✅ Ver comparación de los 5 modelos entrenados"):
        comp = pd.DataFrame(META["comparacion"])[
            ["Modelo", "Accuracy", "F1-Score", "Precision", "Recall", "AUC"]]
        st.dataframe(
            comp.style.format({c: "{:.4f}" for c in
                               ["Accuracy", "F1-Score", "Precision", "Recall", "AUC"]})
                .apply(lambda r: ['background-color:#143d2b' if r["Modelo"] == "XGBoost"
                                  else '' for _ in r], axis=1),
            use_container_width=True, hide_index=True)
        st.caption("Se seleccionó **XGBoost** como modelo final por su mayor AUC (capacidad "
                   "de discriminación) y buen Recall de la clase positiva, prioritario en un "
                   "contexto clínico donde no detectar a un paciente en riesgo es más costoso "
                   "que una falsa alarma.")
