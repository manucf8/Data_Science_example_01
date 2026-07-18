
"""
Dashboard COVID-19 (datos sintéticos) - Streamlit + Plotly
-----------------------------------------------------------
Genera datos sintéticos dentro de la propia app (10,000 registros / 8 columnas),
calcula métricas cuantitativas y cualitativas, y permite construir gráficas
dinámicas eligiendo variables, umbrales y opciones de personalización.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID-19 (Datos Sintéticos)",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIONES = [
    "Antioquia", "Bogotá D.C.", "Valle del Cauca", "Atlántico", "Santander",
    "Cundinamarca", "Bolívar", "Nariño", "Córdoba", "Tolima",
]
REGION_PESOS = [0.20, 0.22, 0.14, 0.09, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04]


# ----------------------------------------------------------------------------
# GENERACIÓN DE DATOS SINTÉTICOS (8 columnas, tipos variados)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # 1) fecha_reporte -> datetime
    fecha_inicio = datetime(2020, 3, 6)
    dias_rango = 900
    offsets = rng.integers(0, dias_rango, n_registros)
    fecha_reporte = pd.to_datetime([fecha_inicio + timedelta(days=int(d)) for d in offsets])

    # 2) region -> categórica (str)
    region = rng.choice(REGIONES, size=n_registros, p=REGION_PESOS)

    # 3) edad -> int (distribución gamma truncada, más realista que uniforme)
    edad = rng.gamma(shape=2.2, scale=17, size=n_registros)
    edad = np.clip(edad, 0, 95).astype(int)

    # 4) sexo -> categórica binaria
    sexo = rng.choice(["Femenino", "Masculino"], size=n_registros, p=[0.51, 0.49])

    # 5) comorbilidad -> booleano (más probable con edad avanzada)
    prob_comorb = np.clip(0.10 + (edad / 95) * 0.55, 0.05, 0.85)
    comorbilidad = rng.random(n_registros) < prob_comorb

    # 6) dias_sintomas -> int (Poisson)
    dias_sintomas = rng.poisson(lam=7, size=n_registros)
    dias_sintomas = np.clip(dias_sintomas, 0, 45)

    # 7) temperatura_corporal -> float
    temperatura = rng.normal(loc=37.6 + (dias_sintomas > 10) * 0.4, scale=0.9, size=n_registros)
    temperatura = np.round(np.clip(temperatura, 35.0, 41.5), 1)

    # 8) estado_paciente -> categórica ordinal, dependiente de edad/comorbilidad
    riesgo = np.clip(0.02 + (edad / 95) * 0.28 + comorbilidad * 0.12, 0.02, 0.55)
    estado_paciente = []
    for r in riesgo:
        p_fallecido = r * 0.30
        p_activo = 0.15
        p_recuperado = 1 - p_fallecido - p_activo
        estado_paciente.append(
            rng.choice(["Activo", "Recuperado", "Fallecido"],
                       p=[p_activo, p_recuperado, p_fallecido])
        )

    df = pd.DataFrame({
        "fecha_reporte": fecha_reporte,
        "region": region,
        "edad": edad,
        "sexo": sexo,
        "estado_paciente": estado_paciente,
        "dias_sintomas": dias_sintomas,
        "temperatura_corporal": temperatura,
        "comorbilidad": comorbilidad,
    })

    return df.sort_values("fecha_reporte").reset_index(drop=True)


# ----------------------------------------------------------------------------
# SIDEBAR - CONTROLES DE GENERACIÓN Y FILTROS
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Generación de datos")

n_registros = st.sidebar.slider(
    "Número de registros", min_value=2_000, max_value=20_000, value=10_000, step=1_000
)
semilla = st.sidebar.number_input("Semilla aleatoria (reproducibilidad)", value=42, step=1)

if st.sidebar.button("🔄 Regenerar datos sintéticos"):
    generar_datos.clear()

df = generar_datos(n_registros, int(semilla))

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")

fecha_min, fecha_max = df["fecha_reporte"].min(), df["fecha_reporte"].max()
rango_fechas = st.sidebar.date_input(
    "Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max
)
regiones_sel = st.sidebar.multiselect("Región", options=REGIONES, default=REGIONES)
sexo_sel = st.sidebar.multiselect("Sexo", options=df["sexo"].unique().tolist(),
                                   default=df["sexo"].unique().tolist())
estado_sel = st.sidebar.multiselect("Estado del paciente",
                                     options=df["estado_paciente"].unique().tolist(),
                                     default=df["estado_paciente"].unique().tolist())
comorb_sel = st.sidebar.multiselect("Comorbilidad", options=[True, False], default=[True, False],
                                     format_func=lambda x: "Sí" if x else "No")

if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    f_ini, f_fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
else:
    f_ini, f_fin = fecha_min, fecha_max

df_f = df[
    (df["fecha_reporte"] >= f_ini) & (df["fecha_reporte"] <= f_fin) &
    (df["region"].isin(regiones_sel)) &
    (df["sexo"].isin(sexo_sel)) &
    (df["estado_paciente"].isin(estado_sel)) &
    (df["comorbilidad"].isin(comorb_sel))
].copy()

st.sidebar.markdown(f"**Registros filtrados:** {len(df_f):,} / {len(df):,}")

COLS_NUM = ["edad", "dias_sintomas", "temperatura_corporal"]
COLS_CAT = ["region", "sexo", "estado_paciente", "comorbilidad"]

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🦠 Dashboard COVID-19 — Datos Sintéticos")
st.caption(
    "Todos los datos se generan de forma sintética dentro de la aplicación con fines "
    "demostrativos; no representan casos reales."
)

if df_f.empty:
    st.warning("No hay registros con los filtros actuales. Ajusta los filtros en la barra lateral.")
    st.stop()

tab_resumen, tab_cuant, tab_cual, tab_viz = st.tabs(
    ["📋 Resumen", "📈 Estadísticas Cuantitativas", "🗂️ Estadísticas Cualitativas", "🎛️ Visualización Dinámica"]
)

# ----------------------------------------------------------------------------
# TAB 1: RESUMEN
# ----------------------------------------------------------------------------
with tab_resumen:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total registros", f"{len(df_f):,}")
    c2.metric("Edad promedio", f"{df_f['edad'].mean():.1f} años")
    tasa_fallec = (df_f["estado_paciente"] == "Fallecido").mean() * 100
    c3.metric("Tasa de fallecidos", f"{tasa_fallec:.1f}%")
    tasa_recup = (df_f["estado_paciente"] == "Recuperado").mean() * 100
    c4.metric("Tasa de recuperados", f"{tasa_recup:.1f}%")
    tasa_comorb = df_f["comorbilidad"].mean() * 100
    c5.metric("Con comorbilidad", f"{tasa_comorb:.1f}%")

    st.markdown("#### Vista previa de los datos")
    st.dataframe(df_f.head(50), use_container_width=True)

    st.markdown("#### Tipos de datos por columna")
    tipos = pd.DataFrame({
        "Columna": df_f.columns,
        "Tipo de dato": df_f.dtypes.astype(str).values,
        "Valores únicos": [df_f[c].nunique() for c in df_f.columns],
        "Nulos": [df_f[c].isna().sum() for c in df_f.columns],
    })
    st.dataframe(tipos, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 2: ESTADÍSTICAS CUANTITATIVAS
# ----------------------------------------------------------------------------
with tab_cuant:
    st.markdown("### Estadística descriptiva — variables numéricas")
    st.dataframe(df_f[COLS_NUM].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("### Distribución de una variable numérica")
    var_num = st.selectbox("Variable numérica", COLS_NUM, key="var_num_dist")
    col_a, col_b = st.columns(2)

    with col_a:
        n_bins = st.slider("Número de bins", 5, 100, 30, key="bins_dist")
        fig_hist = px.histogram(df_f, x=var_num, nbins=n_bins, color="estado_paciente",
                                 title=f"Histograma de {var_num}", opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(df_f, y=var_num, x="sexo", color="sexo", points="outliers",
                          title=f"Boxplot de {var_num} por sexo")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Matriz de correlación (variables numéricas)")
    corr = df_f[COLS_NUM].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Correlación entre variables numéricas")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: ESTADÍSTICAS CUALITATIVAS
# ----------------------------------------------------------------------------
with tab_cual:
    st.markdown("### Frecuencias de variables categóricas")
    var_cat = st.selectbox("Variable categórica", COLS_CAT, key="var_cat_freq")

    conteo = df_f[var_cat].value_counts().reset_index()
    conteo.columns = [var_cat, "conteo"]
    conteo["porcentaje"] = (conteo["conteo"] / conteo["conteo"].sum() * 100).round(2)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(conteo, use_container_width=True, hide_index=True)
        moda = df_f[var_cat].mode().iloc[0]
        st.metric("Moda", str(moda))

    with col_b:
        tipo_grafico_cat = st.radio("Tipo de gráfica", ["Barras", "Pastel"], horizontal=True)
        if tipo_grafico_cat == "Barras":
            fig_cat = px.bar(conteo, x=var_cat, y="conteo", color=var_cat, text="porcentaje",
                              title=f"Distribución de {var_cat}")
        else:
            fig_cat = px.pie(conteo, names=var_cat, values="conteo", title=f"Distribución de {var_cat}", hole=0.35)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### Tabla cruzada (dos variables categóricas)")
    col_c, col_d = st.columns(2)
    with col_c:
        var_cat_1 = st.selectbox("Variable 1", COLS_CAT, index=0, key="cross1")
    with col_d:
        var_cat_2 = st.selectbox("Variable 2", COLS_CAT, index=2, key="cross2")

    if var_cat_1 != var_cat_2:
        tabla_cruzada = pd.crosstab(df_f[var_cat_1], df_f[var_cat_2])
        st.dataframe(tabla_cruzada, use_container_width=True)
        fig_stack = px.bar(tabla_cruzada, barmode="stack", title=f"{var_cat_1} vs {var_cat_2}")
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Elige dos variables distintas para la tabla cruzada.")

# ----------------------------------------------------------------------------
# TAB 4: VISUALIZACIÓN DINÁMICA E INTERACTIVA
# ----------------------------------------------------------------------------
with tab_viz:
    st.markdown("### Constructor de gráficas dinámico")

    todas_cols = df_f.columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        tipo_grafico = st.selectbox(
            "Tipo de gráfica",
            ["Dispersión (scatter)", "Línea", "Barras", "Boxplot", "Violín", "Histograma"],
        )
    with col2:
        eje_x = st.selectbox("Variable eje X", todas_cols, index=todas_cols.index("fecha_reporte"))
    with col3:
        eje_y_opciones = ["(ninguna)"] + todas_cols
        eje_y = st.selectbox("Variable eje Y", eje_y_opciones,
                              index=eje_y_opciones.index("temperatura_corporal")
                              if "temperatura_corporal" in eje_y_opciones else 0)

    col4, col5, col6 = st.columns(3)
    with col4:
        color_por = st.selectbox("Colorear por", ["(ninguna)"] + COLS_CAT)
    with col5:
        paleta = st.selectbox(
            "Paleta de colores",
            ["Plotly", "Vivid", "Bold", "Pastel", "Set2", "D3"],
        )
    with col6:
        opacidad = st.slider("Opacidad", 0.1, 1.0, 0.8)

    color_arg = None if color_por == "(ninguna)" else color_por
    y_arg = None if eje_y == "(ninguna)" else eje_y
    mapa_paletas = {
        "Plotly": px.colors.qualitative.Plotly, "Vivid": px.colors.qualitative.Vivid,
        "Bold": px.colors.qualitative.Bold, "Pastel": px.colors.qualitative.Pastel,
        "Set2": px.colors.qualitative.Set2, "D3": px.colors.qualitative.D3,
    }

    titulo_personalizado = st.text_input("Título de la gráfica", value=f"{eje_x} vs {eje_y}")

    fig = None
    try:
        if tipo_grafico == "Dispersión (scatter)" and y_arg:
            fig = px.scatter(df_f, x=eje_x, y=y_arg, color=color_arg, opacity=opacidad,
                              color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        elif tipo_grafico == "Línea" and y_arg:
            df_linea = df_f.sort_values(eje_x)
            fig = px.line(df_linea, x=eje_x, y=y_arg, color=color_arg,
                           color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        elif tipo_grafico == "Barras" and y_arg:
            fig = px.bar(df_f, x=eje_x, y=y_arg, color=color_arg, opacity=opacidad,
                         color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        elif tipo_grafico == "Boxplot" and y_arg:
            fig = px.box(df_f, x=eje_x, y=y_arg, color=color_arg,
                        color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        elif tipo_grafico == "Violín" and y_arg:
            fig = px.violin(df_f, x=eje_x, y=y_arg, color=color_arg, box=True,
                            color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        elif tipo_grafico == "Histograma":
            fig = px.histogram(df_f, x=eje_x, color=color_arg, opacity=opacidad,
                               color_discrete_sequence=mapa_paletas[paleta], title=titulo_personalizado)
        else:
            st.info("Selecciona una variable Y válida para este tipo de gráfica.")
    except Exception as e:
        st.error(f"No fue posible construir la gráfica con esta combinación de variables: {e}")

    # -------------------- Umbrales / líneas de referencia --------------------
    st.markdown("#### Umbrales de referencia (líneas horizontales/verticales)")
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        activar_umbral_y = st.checkbox("Activar umbral en eje Y (numérico)")
        if activar_umbral_y and y_arg in COLS_NUM and fig is not None:
            min_y, max_y = float(df_f[y_arg].min()), float(df_f[y_arg].max())
            umbral_y = st.slider(f"Umbral para {y_arg}", min_y, max_y, (min_y + max_y) / 2)
            fig.add_hline(y=umbral_y, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral {y_arg}: {umbral_y:.2f}")
            n_sobre = (df_f[y_arg] > umbral_y).sum()
            st.caption(f"Registros por encima del umbral: **{n_sobre:,}** ({n_sobre/len(df_f)*100:.1f}%)")

    with col_u2:
        activar_umbral_x = st.checkbox("Activar umbral en eje X (numérico)")
        if activar_umbral_x and eje_x in COLS_NUM and fig is not None:
            min_x, max_x = float(df_f[eje_x].min()), float(df_f[eje_x].max())
            umbral_x = st.slider(f"Umbral para {eje_x}", min_x, max_x, (min_x + max_x) / 2)
            fig.add_vline(x=umbral_x, line_dash="dash", line_color="blue",
                          annotation_text=f"Umbral {eje_x}: {umbral_x:.2f}")
            n_sobre_x = (df_f[eje_x] > umbral_x).sum()
            st.caption(f"Registros por encima del umbral: **{n_sobre_x:,}** ({n_sobre_x/len(df_f)*100:.1f}%)")

    if fig is not None:
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Descargar datos filtrados")
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV filtrado", data=csv, file_name="covid_datos_sinteticos_filtrados.csv",
                       mime="text/csv")
