import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CardioRiesgo | Análisis de ataque al corazón",
                   page_icon=":material/monitor_heart:", layout="wide")

VINO = "#7A1F2B"
VINO_OSC = "#5C1520"
ROJO = "#B03A48"
ROSA = "#F3E3E6"
ROSA_MED = "#D9B8BE"
GRIS = "#6B7280"
BORDE = "#E7E3E4"
FONDO = "#F7F6F6"
TINTA = "#1F1A1B"
VERDE = "#2F7D5B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select, label {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
.stApp {{ background: {FONDO}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 1280px; }}

/* Barra lateral */
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDE}; }}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
.marca {{ display: flex; align-items: center; gap: .7rem; padding: 0 .2rem 1.2rem .2rem; border-bottom: 1px solid {BORDE}; margin-bottom: 1.1rem; }}
.marca .logo {{ width: 38px; height: 38px; border-radius: 10px; background: {VINO}; color: #fff; display: flex;
    align-items: center; justify-content: center; font-weight: 700; font-size: 1rem; letter-spacing: .02em; }}
.marca .nom {{ font-weight: 700; color: {TINTA}; font-size: 1.05rem; line-height: 1.1; }}
.marca .sub {{ color: {GRIS}; font-size: .75rem; }}
.seccion-menu {{ color: {GRIS}; font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin: .2rem 0 .5rem .3rem; }}
section[data-testid="stSidebar"] .stButton {{ margin-bottom: -.35rem; }}
section[data-testid="stSidebar"] .stButton button {{
    justify-content: flex-start; text-align: left; border: none; border-radius: 8px;
    padding: .55rem .8rem; font-weight: 500; font-size: .92rem; box-shadow: none; transition: background .15s, color .15s;
}}
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {{ background: transparent; color: #3D3537; }}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {{ background: {ROSA}; color: {VINO}; }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{ background: {ROSA}; color: {VINO}; font-weight: 600;
    box-shadow: inset 3px 0 0 {VINO}; }}
section[data-testid="stSidebar"] .stButton button p {{ font-size: .92rem; }}
section[data-testid="stSidebar"] .stButton button > div {{ justify-content: flex-start; width: 100%; }}
section[data-testid="stSidebar"] .stButton button div[data-testid="stMarkdownContainer"] {{ text-align: left; }}
.st-key-gauge div[data-testid="stPlotlyChart"] {{ border: none; padding: 0; }}
div[data-testid="stRadio"] label p {{ font-size: .9rem; }}
.pie-menu {{ color: {GRIS}; font-size: .78rem; border-top: 1px solid {BORDE}; padding-top: .9rem; margin-top: 1.2rem; }}

/* Botones del área principal */
div[data-testid="stMainBlockContainer"] .stButton button,
div[data-testid="stMainBlockContainer"] .stDownloadButton button,
div[data-testid="stFormSubmitButton"] button {{
    border-radius: 8px; font-weight: 600; font-size: .9rem; padding: .6rem 1.1rem; min-height: 2.6rem;
    transition: background .15s, border-color .15s, box-shadow .15s;
}}
div[data-testid="stMainBlockContainer"] .stButton button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {{
    background: {VINO}; color: #fff; border: 1px solid {VINO}; box-shadow: 0 1px 2px rgba(0,0,0,.08);
}}
div[data-testid="stMainBlockContainer"] .stButton button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {{ background: {VINO_OSC}; border-color: {VINO_OSC}; color: #fff;
    box-shadow: 0 4px 12px rgba(122,31,43,.25); }}
div[data-testid="stMainBlockContainer"] .stButton button[kind="secondary"],
div[data-testid="stMainBlockContainer"] .stDownloadButton button {{
    background: #fff; color: {TINTA}; border: 1px solid {BORDE};
}}
div[data-testid="stMainBlockContainer"] .stButton button[kind="secondary"]:hover,
div[data-testid="stMainBlockContainer"] .stDownloadButton button:hover {{ border-color: {VINO}; color: {VINO}; background: #fff; }}


/* Encabezado de página */
.encabezado {{ background: linear-gradient(135deg, #4A1621 0%, #2A0C12 100%); border-radius: 12px; padding: 22px 28px;
    margin-bottom: 1.4rem; display: flex; justify-content: space-between; align-items: center; gap: 1rem; }}
.encabezado .migas {{ color: #D9B8BE; font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; }}
.encabezado h1 {{ color: #FFFFFF; font-size: 1.6rem; font-weight: 800; margin: .25rem 0 0 0; padding: 0; }}
.encabezado p {{ color: #E6CDD1; margin: .3rem 0 0 0; font-size: .9rem; }}
.etiqueta {{ background: rgba(255,255,255,.12); color: #F3E3E6; border: 1px solid rgba(255,255,255,.22);
    padding: .35rem .8rem; border-radius: 20px; font-size: .75rem; font-weight: 600; white-space: nowrap; }}
.alerta {{ background: #fff; border: 1px solid {BORDE}; border-left: 4px solid {VINO}; border-radius: 10px;
    padding: .85rem 1.1rem; margin-bottom: .6rem; }}
.alerta .t {{ font-weight: 700; color: {TINTA}; font-size: .9rem; }}
.alerta .d {{ color: {GRIS}; font-size: .82rem; margin-top: .15rem; }}
.alerta.ambar {{ border-left-color: #C98A2B; }}
.alerta.gris {{ border-left-color: #9CA3AF; }}

/* Selector segmentado tipo píldora */
div[data-testid="stButtonGroup"] {{ background: #F1ECED; padding: 5px; border-radius: 10px; display: inline-flex; }}
div[data-testid="stButtonGroup"] button {{ border: none !important; background: transparent; color: #5A4A4D;
    font-weight: 600; font-size: .85rem; border-radius: 8px !important; padding: .4rem 1rem; }}
div[data-testid="stButtonGroup"] button[aria-checked="true"], div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {{
    background: {VINO} !important; color: #fff !important; }}
div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p {{ color: #fff !important; }}

/* Tarjetas */
.kpi {{ background: #fff; border: 1px solid {BORDE}; border-top: 4px solid {ROSA_MED}; border-radius: 12px; padding: 1rem 1.2rem; height: 100%; }}
.kpi .lbl {{ color: {GRIS}; font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.kpi .val {{ color: {VINO}; font-size: 1.7rem; font-weight: 800; line-height: 1.25; margin-top: .25rem; }}
.kpi .sub {{ color: {GRIS}; font-size: .8rem; }}
.kpi.destacado {{ border-top-color: {VINO}; }}
.subtitulo {{ color: {TINTA}; font-weight: 600; font-size: 1.02rem; margin: 1.4rem 0 .6rem 0; }}
.nota {{ background: #fff; border: 1px solid {BORDE}; border-left: 3px solid {VINO}; border-radius: 8px;
    padding: .8rem 1rem; color: #3D3537; font-size: .9rem; margin: .4rem 0 1rem 0; }}
div[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {BORDE}; border-radius: 10px; padding: .4rem .6rem; }}
div[data-testid="stForm"] {{ background: #fff; border: 1px solid {BORDE}; border-radius: 10px; padding: 1.2rem 1.3rem; }}
div[data-testid="stExpander"] details {{ background: #fff; border: 1px solid {BORDE}; border-radius: 10px; }}
.resultado {{ background: #fff; border: 1px solid {BORDE}; border-radius: 10px; padding: 1.3rem 1.4rem; height: 100%; }}
.resultado .lbl {{ color: {GRIS}; font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.estado {{ display: inline-block; padding: .35rem .8rem; border-radius: 6px; font-weight: 600; font-size: .85rem; margin: .5rem 0 .7rem 0; }}
.estado.alto {{ background: #FBE7EA; color: {ROJO}; border: 1px solid #F0C4CB; }}
.estado.bajo {{ background: #E5F2EC; color: {VERDE}; border: 1px solid #C3E0D2; }}
.aviso {{ color: {GRIS}; font-size: .78rem; margin-top: .8rem; }}
</style>
""", unsafe_allow_html=True)

RENOMBRAR = {
    "age": "edad", "hypertension": "hipertension", "heart_disease": "cardiopatia",
    "ever_married": "casado", "avg_glucose_level": "glucosa_promedio",
    "smoking_status": "tabaquismo", "stroke_ataque_corazon": "ataque_corazon",
}
SI_NO = {"Yes": "Sí", "No": "No"}
TABACO = {"never smoked": "Nunca ha fumado", "formerly smoked": "Exfumador",
          "smokes": "Fumador", "Unknown": "Desconocido"}
ORDEN_TABACO = ["Nunca ha fumado", "Exfumador", "Fumador", "Desconocido"]
VARS = ["edad", "hipertension", "cardiopatia", "casado", "glucosa_promedio", "tabaquismo"]


def limpiar(df):
    df = df.rename(columns=RENOMBRAR).copy()
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] =df[c].astype(str).str.strip().str.strip("'").str.strip()
    for c in ["hipertension", "cardiopatia", "casado", "ataque_corazon"]:
        if c in df.columns:
            df[c] = df[c].map(SI_NO).fillna(df[c])
    df["tabaquismo"] = df["tabaquismo"].map(TABACO).fillna(df["tabaquismo"])
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["glucosa_promedio"] = pd.to_numeric(df["glucosa_promedio"], errors="coerce")
    return df


@st.cache_data
def cargar(origen):
    datos = pd.read_excel(origen, sheet_name="Datos")
    try:
        futuros = pd.read_excel(origen, sheet_name="Predicciones futuras")
    except ValueError:
        futuros = None
    original = datos.copy()
    df = limpiar(datos)
    n0 = len(df)
    nulos = int(df.isna().sum().sum())
    df = df.dropna()
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    fuera = int(((df["edad"] < 0) | (df["edad"] > 120) | (df["glucosa_promedio"] <= 0)).sum())
    df = df[(df["edad"].between(0, 120)) & (df["glucosa_promedio"] > 0)].reset_index(drop=True)
    df["grupo_edad"] = pd.cut(df["edad"], [0, 18, 30, 45, 60, 75, 120],
                              labels=["0-17", "18-29", "30-44", "45-59", "60-74", "75+"], right=False)
    df["nivel_glucosa"] = pd.cut(df["glucosa_promedio"], [0, 100, 126, 1000],
                                 labels=["Normal (<100)", "Prediabetes (100-125)", "Diabetes (≥126)"], right=False)
    resumen = {"filas_originales": n0, "nulos": nulos, "duplicados": dup, "fuera_rango": fuera, "filas_finales": len(df)}
    if futuros is not None:
        futuros = limpiar(futuros)
    return original, df, futuros, resumen


def a_modelo(df, columnas=None):
    X = df[VARS].copy()
    for c in ["hipertension", "cardiopatia", "casado"]:
        X[c] = (X[c] == "Sí").astype(int)
    X["tabaquismo"] = pd.Categorical(X["tabaquismo"], categories=ORDEN_TABACO)
    X = pd.get_dummies(X, columns=["tabaquismo"], drop_first=True).astype(float)
    if columnas is not None:
        X = X.reindex(columns=columnas, fill_value=0)
    return X


@st.cache_resource
def entrenar(df, nombre):
    X = a_modelo(df)
    y = (df["ataque_corazon"] == "Sí").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    if nombre == "Regresión logística":
        m = LogisticRegression(class_weight="balanced", max_iter=3000)
    else:
        m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    met = {"Exactitud": accuracy_score(yte, pred), "Sensibilidad": recall_score(yte, pred),
           "Precisión": precision_score(yte, pred, zero_division=0), "F1": f1_score(yte, pred),
           "AUC": roc_auc_score(yte, prob)}
    if nombre == "Regresión logística":
        imp = pd.Series(m.coef_[0], index=X.columns)
    else:
        imp = pd.Series(m.feature_importances_, index=X.columns)
    return m, list(X.columns), met, confusion_matrix(yte, pred), roc_curve(yte, prob), imp


def kpi(col, etiqueta, valor, sub="", destacado=False):
    clase = "kpi destacado" if destacado else "kpi"
    col.markdown(f'<div class="{clase}"><div class="lbl">{etiqueta}</div><div class="val">{valor}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def encabezado(migas, titulo, descripcion, etiqueta=""):
    tag = f'<span class="etiqueta">{etiqueta}</span>' if etiqueta else ""
    st.markdown(f'<div class="encabezado"><div><div class="migas">{migas}</div><h1>{titulo}</h1>'
                f'<p>{descripcion}</p></div>{tag}</div>', unsafe_allow_html=True)


def subtitulo(texto):
    st.markdown(f'<div class="subtitulo">{texto}</div>', unsafe_allow_html=True)


def miles(n):
    return f"{n:,}".replace(",", ".")


def estilo(fig, alto=360, titulo=None):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=48 if titulo else 20, b=10),
                      font=dict(family="Inter, sans-serif", color="#3D3537", size=12),
                      title=dict(text=titulo, font=dict(size=14, color=TINTA), x=0.01) if titulo else None,
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", legend_title_text="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.97))
    fig.update_xaxes(showgrid=False, linecolor=BORDE)
    fig.update_yaxes(gridcolor="#F0EDEE", zeroline=False)
    return fig


def tasa(df, col):
    t = df.groupby(col, observed=True)["ataque_corazon"].apply(lambda s: (s == "Sí").mean() * 100).reset_index()
    t.columns = [col, "tasa"]
    return t


def medidor(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": " %", "valueformat": ".1f", "font": {"size": 40, "color": TINTA}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDE},
               "bar": {"color": VINO, "thickness": 0.28}, "bgcolor": "#fff", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "#EAF3EE"}, {"range": [30, 50], "color": "#FAF1E3"},
                         {"range": [50, 100], "color": "#F8E6E9"}],
               "threshold": {"line": {"color": TINTA, "width": 2}, "value": 50}}))
    fig = estilo(fig, 280)
    fig.update_layout(margin=dict(l=55, r=55, t=30, b=5))
    return fig


COLOR_CLASE = {"No": ROSA_MED, "Sí": VINO}

# ---------------- carga ----------------
ruta = Path(__file__).parent / "ataque_corazon.xlsx"
if ruta.exists():
    origen = str(ruta)
else:
    archivo = st.sidebar.file_uploader("Cargar ataque_corazon.xlsx", type=["xlsx"])
    if archivo is None:
        st.info("Cargue el archivo ataque_corazon.xlsx en el panel lateral para comenzar.")
        st.stop()
    origen = archivo
original, df, futuros, resumen = cargar(origen)

for clave, valor in {"pagina": "Resumen", "modelo": "Regresión logística", "factor": "Edad"}.items():
    st.session_state.setdefault(clave, valor)

MENU = [("Resumen", ":material/space_dashboard:"), ("Limpieza de datos", ":material/cleaning_services:"),
        ("Exploración", ":material/bar_chart:"), ("Modelo", ":material/model_training:"),
        ("Predicción", ":material/monitor_heart:")]

with st.sidebar:
    st.markdown('<div class="marca"><div class="logo">CR</div><div><div class="nom">CardioRiesgo</div>'
                '<div class="sub">Análisis de ataque al corazón</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="seccion-menu">Navegación</div>', unsafe_allow_html=True)
    for nombre, icono in MENU:
        activo = st.session_state.pagina == nombre
        if st.button(nombre, key=f"nav_{nombre}", icon=icono, width="stretch",
                     type="primary" if activo else "secondary"):
            st.session_state.pagina = nombre
            st.rerun()
    st.markdown(f'<div class="pie-menu">Registros analizados: <b>{miles(len(df))}</b><br>'
                f'Modelo activo: <b>{st.session_state.modelo}</b></div>', unsafe_allow_html=True)

pagina = st.session_state.pagina
casos_total = int((df["ataque_corazon"] == "Sí").sum())

# ---------------- RESUMEN ----------------
if pagina == "Resumen":
    encabezado("Panel general", "Riesgo de ataque al corazón",
               "Factores clínicos asociados y modelo predictivo sobre la base de pacientes.", f"{miles(len(df))} pacientes")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Pacientes", miles(len(df)), "registros después de la limpieza", True)
    kpi(c2, "Casos de ataque", casos_total, f"{casos_total / len(df) * 100:.1f} % del total")
    kpi(c3, "Edad promedio", f"{df['edad'].mean():.1f}", "años")
    kpi(c4, "Glucosa promedio", f"{df['glucosa_promedio'].mean():.1f}", "mg/dL")
    st.write("")
    a, b = st.columns([1, 1.5])
    with a:
        conteo = df["ataque_corazon"].value_counts().reindex(["No", "Sí"]).reset_index()
        conteo.columns = ["Ataque", "Pacientes"]
        fig = px.pie(conteo, names="Ataque", values="Pacientes", hole=0.68, color="Ataque", color_discrete_map=COLOR_CLASE)
        fig.update_layout(showlegend=False)
        fig.update_traces(textinfo="label+percent", textfont_size=12, sort=False,
                          marker=dict(line=dict(color="#fff", width=2)))
        fig.add_annotation(text=f"<b>{casos_total / len(df) * 100:.1f} %</b><br><span style='font-size:11px'>con ataque</span>",
                           showarrow=False, font=dict(size=20, color=TINTA))
        st.plotly_chart(estilo(fig, 360, "Distribución de pacientes"), width="stretch")
    with b:
        t = tasa(df, "grupo_edad")
        fig = px.bar(t, x="grupo_edad", y="tasa", text=t["tasa"].map(lambda v: f"{v:.1f} %"),
                     color_discrete_sequence=[VINO])
        fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
        fig.update_layout(xaxis_title="Grupo de edad", yaxis_title="% con ataque", bargap=0.35)
        st.plotly_chart(estilo(fig, 360, "Tasa de ataque por grupo de edad"), width="stretch")
    subtitulo("Hallazgos principales")
    edad_alta = tasa(df, "grupo_edad").set_index("grupo_edad")["tasa"]
    t_hip = tasa(df, "hipertension").set_index("hipertension")["tasa"]
    t_car = tasa(df, "cardiopatia").set_index("cardiopatia")["tasa"]
    t_glu = tasa(df, "nivel_glucosa").set_index("nivel_glucosa")["tasa"]
    h1, h2 = st.columns(2)
    h1.markdown(f'<div class="alerta"><div class="t">Edad: factor de mayor peso</div><div class="d">Los mayores de 75 años '
                f'presentan una tasa de {edad_alta.iloc[-1]:.1f} %, frente a {edad_alta.iloc[0]:.1f} % en menores de 18.</div></div>'
                f'<div class="alerta"><div class="t">Cardiopatía previa</div><div class="d">Tasa de {t_car["Sí"]:.1f} % en pacientes '
                f'con cardiopatía y {t_car["No"]:.1f} % sin ella.</div></div>', unsafe_allow_html=True)
    h2.markdown(f'<div class="alerta ambar"><div class="t">Hipertensión</div><div class="d">Tasa de {t_hip["Sí"]:.1f} % en '
                f'hipertensos frente a {t_hip["No"]:.1f} % en no hipertensos.</div></div>'
                f'<div class="alerta gris"><div class="t">Glucosa elevada</div><div class="d">Con glucosa ≥126 mg/dL la tasa es '
                f'{t_glu.iloc[-1]:.1f} %; con glucosa normal, {t_glu.iloc[0]:.1f} %.</div></div>', unsafe_allow_html=True)
    subtitulo("Accesos rápidos")
    cols = st.columns(4)
    accesos = [("Ver limpieza", "Limpieza de datos", ":material/cleaning_services:"),
               ("Explorar datos", "Exploración", ":material/bar_chart:"),
               ("Evaluar modelo", "Modelo", ":material/model_training:"),
               ("Calcular riesgo", "Predicción", ":material/arrow_forward:")]
    for i, (col, (txt, dest, ic)) in enumerate(zip(cols, accesos)):
        if col.button(txt, key=f"ir_{dest}", icon=ic, width="stretch", type="primary" if i == 3 else "secondary"):
            st.session_state.pagina = dest
            st.rerun()

# ---------------- LIMPIEZA ----------------
elif pagina == "Limpieza de datos":
    encabezado("Preparación", "Limpieza de datos", "Transformaciones aplicadas a la base original antes del análisis.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Filas originales", miles(resumen["filas_originales"]))
    kpi(c2, "Valores nulos", resumen["nulos"])
    kpi(c3, "Duplicados eliminados", resumen["duplicados"])
    kpi(c4, "Filas finales", miles(resumen["filas_finales"]), destacado=True)
    subtitulo("Pasos de limpieza")
    pasos = pd.DataFrame({
        "Paso": ["Nombres de columnas", "Texto de tabaquismo", "Valores Yes / No", "Categorías de tabaquismo",
                 "Valores nulos", "Registros duplicados", "Rangos de edad y glucosa", "Variables derivadas"],
        "Acción": ["Traducción al español", "Eliminación de comillas sobrantes",
                   "Conversión a Sí / No", "Traducción: Nunca ha fumado, Exfumador, Fumador, Desconocido",
                   f"Revisión: {resumen['nulos']} encontrados", f"Eliminación de {resumen['duplicados']} filas repetidas",
                   f"Validación: {resumen['fuera_rango']} fuera de rango",
                   "Creación de grupo de edad y nivel de glucosa"]})
    st.dataframe(pasos, hide_index=True, width="stretch")
    subtitulo("Vista de los datos")
    vista = st.segmented_control("Vista", ["Datos limpios", "Datos originales"], default="Datos limpios",
                                 label_visibility="collapsed")
    if vista == "Datos originales":
        st.dataframe(original, height=330, width="stretch")
    else:
        st.dataframe(df.drop(columns=["grupo_edad", "nivel_glucosa"]), height=330, width="stretch")
    subtitulo("Estadísticas descriptivas")
    st.dataframe(df[["edad", "glucosa_promedio"]].describe().T.round(2), width="stretch")
    limpio = df.drop(columns=["grupo_edad", "nivel_glucosa"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        limpio.to_excel(w, sheet_name="Datos limpios", index=False)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Descargar Excel", buf.getvalue(), "ataque_corazon_limpio.xlsx",
                       icon=":material/download:", width="stretch")
    d2.download_button("Descargar CSV", limpio.to_csv(index=False).encode("utf-8-sig"), "ataque_corazon_limpio.csv",
                       "text/csv", icon=":material/download:", width="stretch")

# ---------------- EXPLORACIÓN ----------------
elif pagina == "Exploración":
    encabezado("Análisis descriptivo", "Exploración de los datos",
               "Comparación de la tasa de ataque según cada factor de riesgo.")
    with st.expander("Filtros", expanded=True, icon=":material/tune:"):
        f1, f2, f3 = st.columns(3)
        emin, emax = int(df["edad"].min()), int(df["edad"].max())
        rango = f1.slider("Edad", emin, emax, (emin, emax))
        tab = f2.multiselect("Tabaquismo", ORDEN_TABACO, ORDEN_TABACO)
        hip = f3.multiselect("Hipertensión", ["No", "Sí"], ["No", "Sí"])
    d = df[df["edad"].between(*rango) & df["tabaquismo"].isin(tab) & df["hipertension"].isin(hip)]
    if d.empty:
        st.warning("No hay pacientes que cumplan los filtros seleccionados.")
        st.stop()
    casos = int((d["ataque_corazon"] == "Sí").sum())
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Pacientes filtrados", miles(len(d)))
    kpi(c2, "Casos de ataque", casos)
    kpi(c3, "Tasa de ataque", f"{casos / len(d) * 100:.2f} %", destacado=True)
    subtitulo("Factor de riesgo")
    factores = {"Edad": "grupo_edad", "Hipertensión": "hipertension", "Cardiopatía": "cardiopatia",
                "Tabaquismo": "tabaquismo", "Glucosa": "nivel_glucosa", "Estado civil": "casado"}
    elegido = st.segmented_control("Factor", list(factores), default=st.session_state.factor,
                                   label_visibility="collapsed", key="sel_factor")
    if elegido:
        st.session_state.factor = elegido
    nombre_f = st.session_state.factor
    var = factores[nombre_f]
    a, b = st.columns(2)
    with a:
        t = tasa(d, var)
        fig = px.bar(t, x=var, y="tasa", text=t["tasa"].map(lambda v: f"{v:.1f} %"), color_discrete_sequence=[VINO])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="", yaxis_title="% con ataque", bargap=0.4)
        st.plotly_chart(estilo(fig, 360, f"Tasa de ataque según {nombre_f.lower()}"), width="stretch")
    with b:
        g = d.groupby([var, "ataque_corazon"], observed=True).size().reset_index(name="Pacientes")
        fig = px.bar(g, x=var, y="Pacientes", color="ataque_corazon", barmode="group", color_discrete_map=COLOR_CLASE)
        fig.update_layout(xaxis_title="", bargap=0.3)
        st.plotly_chart(estilo(fig, 360, f"Pacientes según {nombre_f.lower()}"), width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.box(d, x="ataque_corazon", y="glucosa_promedio", color="ataque_corazon", points=False,
                     color_discrete_map=COLOR_CLASE)
        fig.update_layout(xaxis_title="Ataque al corazón", yaxis_title="mg/dL", showlegend=False)
        st.plotly_chart(estilo(fig, 360, "Glucosa promedio según ataque"), width="stretch")
    with b:
        fig = px.histogram(d, x="edad", color="ataque_corazon", nbins=30, barmode="overlay", opacity=0.8,
                           color_discrete_map=COLOR_CLASE)
        fig.update_layout(xaxis_title="Edad", yaxis_title="Pacientes")
        st.plotly_chart(estilo(fig, 360, "Distribución de edad"), width="stretch")
    fig = px.scatter(d.sort_values("ataque_corazon"), x="edad", y="glucosa_promedio", color="ataque_corazon",
                     opacity=0.65, color_discrete_map={"No": "#E2D0D3", "Sí": VINO})
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(xaxis_title="Edad", yaxis_title="Glucosa (mg/dL)")
    st.plotly_chart(estilo(fig, 420, "Edad frente a glucosa promedio"), width="stretch")

# ---------------- MODELO ----------------
elif pagina == "Modelo":
    encabezado("Aprendizaje automático", "Modelo predictivo",
               "Desempeño del clasificador sobre el 25 % de los datos reservado para prueba.", st.session_state.modelo)
    subtitulo("Algoritmo")
    elegido = st.segmented_control("Algoritmo", ["Regresión logística", "Bosque aleatorio"],
                                   default=st.session_state.modelo, label_visibility="collapsed", key="sel_modelo")
    if elegido and elegido != st.session_state.modelo:
        st.session_state.modelo = elegido
        st.rerun()
    nombre = st.session_state.modelo
    modelo, columnas, met, cm, (fpr, tpr, _), imp = entrenar(df, nombre)
    st.markdown(f'<div class="nota">Partición 75 % entrenamiento / 25 % prueba con estratificación. '
                f'Se usan pesos de clase balanceados porque solo el {casos_total / len(df) * 100:.1f} % '
                f'de los pacientes presenta ataque.</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (col, (k, v)) in enumerate(zip(cols, met.items())):
        kpi(col, k, f"{v:.3f}" if k == "AUC" else f"{v * 100:.1f} %", destacado=(k == "AUC"))
    st.write("")
    a, b = st.columns(2)
    with a:
        fig = px.imshow(cm, text_auto=True, x=["Predicho: No", "Predicho: Sí"], y=["Real: No", "Real: Sí"],
                        color_continuous_scale=["#FFFFFF", ROSA, VINO])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textfont_size=17)
        st.plotly_chart(estilo(fig, 360, "Matriz de confusión"), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=VINO, width=2.5), fill="tozeroy",
                                 fillcolor="rgba(122,31,43,0.08)", name="Modelo"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#A8A2A3", dash="dash"), name="Azar"))
        fig.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad")
        st.plotly_chart(estilo(fig, 360, f"Curva ROC  ·  AUC {met['AUC']:.3f}"), width="stretch")
    imp = imp.rename(lambda s: s.replace("_", " ").replace("tabaquismo ", "Tabaquismo: ").capitalize())
    imp = imp.reindex(imp.abs().sort_values().index)
    titulo = "Coeficientes del modelo" if nombre == "Regresión logística" else "Importancia de las variables"
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color=imp.values > 0,
                 color_discrete_map={True: VINO, False: ROSA_MED})
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", bargap=0.35)
    st.plotly_chart(estilo(fig, 360, titulo), width="stretch")

# ---------------- PREDICCIÓN ----------------
elif pagina == "Predicción":
    nombre = st.session_state.modelo
    encabezado("Aplicación del modelo", "Predicción de riesgo",
               "Estimación de la probabilidad de ataque para un paciente nuevo o para los casos del archivo.", nombre)
    modelo, columnas, *_ = entrenar(df, nombre)
    vista = st.segmented_control("Modo", ["Paciente nuevo", "Pacientes del archivo"], default="Paciente nuevo",
                                 label_visibility="collapsed", key="modo_pred")
    st.write("")
    if vista != "Pacientes del archivo":
        izq, der = st.columns([1.15, 1])
        with izq:
            with st.form("paciente", border=False):
                a, b = st.columns(2)
                edad = a.number_input("Edad (años)", 0, 110, 55)
                glucosa = b.number_input("Glucosa promedio (mg/dL)", 40.0, 300.0, 100.0, step=0.5)
                hip = a.radio("Hipertensión", ["No", "Sí"], horizontal=True)
                car = b.radio("Cardiopatía", ["No", "Sí"], horizontal=True)
                cas = a.radio("Ha estado casado", ["No", "Sí"], index=1, horizontal=True)
                tab = b.selectbox("Tabaquismo", ORDEN_TABACO)
                st.write("")
                enviar = st.form_submit_button("Calcular riesgo", icon=":material/monitor_heart:", width="stretch")
        with der:
            with st.container(border=True):
                if enviar:
                    p = pd.DataFrame([{"edad": edad, "hipertension": hip or "No", "cardiopatia": car or "No",
                                       "casado": cas or "No", "glucosa_promedio": glucosa, "tabaquismo": tab}])
                    prob = modelo.predict_proba(a_modelo(p, columnas))[0, 1]
                    alto = prob >= 0.5
                    st.markdown('<div class="kpi" style="border:none;padding:0"><div class="lbl">Probabilidad estimada</div></div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(medidor(prob), width="stretch", key="gauge")
                    texto = ("El perfil se asemeja al de los pacientes que presentaron ataque." if alto
                             else "El perfil se asemeja al de los pacientes que no presentaron ataque.")
                    st.markdown(f'<span class="estado {"alto" if alto else "bajo"}">Riesgo {"alto" if alto else "bajo"}</span>'
                                f'<div>{texto}</div><div class="aviso">Resultado con fines académicos. '
                                'No reemplaza una valoración médica.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kpi" style="border:none;padding:0"><div class="lbl">Resultado</div></div>'
                                '<p style="color:#6B7280;margin-top:.6rem">Complete los datos del paciente y presione '
                                '<b>Calcular riesgo</b> para ver la probabilidad estimada.</p>', unsafe_allow_html=True)
    else:
        if futuros is None:
            st.info("El archivo no contiene la hoja 'Predicciones futuras'.")
        else:
            subtitulo("Pacientes por evaluar")
            st.dataframe(futuros, hide_index=True, width="stretch")
            if st.button("Generar predicciones", icon=":material/play_arrow:", type="primary"):
                r = futuros.copy()
                r["probabilidad"] = modelo.predict_proba(a_modelo(r, columnas))[:, 1]
                r["predicción"] = np.where(r["probabilidad"] >= 0.5, "Sí", "No")
                r.insert(0, "paciente", [f"Paciente {i + 1}" for i in range(len(r))])
                fig = px.bar(r, x="paciente", y="probabilidad", text=r["probabilidad"].map(lambda v: f"{v * 100:.1f} %"),
                             color="predicción", color_discrete_map=COLOR_CLASE)
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.add_hline(y=0.5, line_dash="dash", line_color="#A8A2A3", annotation_text="Umbral 50 %",
                              annotation_position="top left")
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Probabilidad", yaxis_range=[0, 1.05],
                                  bargap=0.5)
                st.plotly_chart(estilo(fig, 360, "Probabilidad estimada de ataque"), width="stretch")
                r["probabilidad"] = r["probabilidad"].map(lambda v: f"{v * 100:.1f} %")
                st.dataframe(r, hide_index=True, width="stretch")
