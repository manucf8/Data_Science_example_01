"""
Dashboard de Accidentes de Tránsito - Medellín (datos sintéticos)
------------------------------------------------------------------
EAFIT 2026 - Ciencia de Datos - Profesor Jose Padilla - Julio 2026

Genera datos sintéticos dentro de la propia app (500 registros / 10 columnas,
incluyendo una serie de tiempo), calcula métricas cuantitativas y cualitativas,
y permite construir gráficas dinámicas eligiendo variables, umbrales y
personalización. Acceso protegido con código.
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
    page_title="Accidentes de Tránsito - Medellín",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CODIGO_ACCESO = "4650"

COMUNAS_MEDELLIN = [
    "Popular", "Santa Cruz", "Manrique", "Aranjuez", "Castilla", "Robledo",
    "Villa Hermosa", "Buenos Aires", "La Candelaria", "Laureles-Estadio",
    "La América", "San Javier", "El Poblado", "Guayabal", "Belén",
]
PESOS_COMUNAS = [0.09, 0.06, 0.07, 0.06, 0.07, 0.06, 0.06, 0.06, 0.10, 0.08,
                 0.05, 0.05, 0.09, 0.05, 0.05]

TIPOS_ACCIDENTE = ["Choque", "Atropello", "Volcamiento", "Caída de ocupante", "Incendio"]
CLASES_VIA = ["Avenida", "Calle", "Carrera", "Autopista", "Glorieta"]
CONDICIONES_CLIMA = ["Seco", "Lluvia", "Niebla", "Nublado"]

# ----------------------------------------------------------------------------
# BARRA LATERAL - IDENTIFICACIÓN INSTITUCIONAL
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EAFIT 2026")
    st.markdown("**Ciencia de Datos**")
    st.markdown("Profesor: Jose Padilla")
    st.markdown("Julio 2026")
    st.markdown("---")

# ----------------------------------------------------------------------------
# CONTROL DE ACCESO
# ----------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if not st.session_state["autenticado"]:
    with st.sidebar:
        codigo_ingresado = st.text_input("🔒 Código de acceso", type="password")
        if st.button("Ingresar"):
            if codigo_ingresado == CODIGO_ACCESO:
                st.session_state["autenticado"] = True
                st.rerun()
            else:
                st.error("Código incorrecto.")
    st.title("🚦 Dashboard de Accidentes de Tránsito - Medellín")
    st.info("Ingresa el código de acceso en la barra lateral para operar el dashboard.")
    st.stop()

# ----------------------------------------------------------------------------
# GENERACIÓN DE DATOS SINTÉTICOS (10 columnas, incluye serie de tiempo)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # 1) fecha -> datetime (serie de tiempo, ~2 años)
    fecha_inicio = datetime(2024, 1, 1)
    dias_rango = 730
    offsets = np.sort(rng.integers(0, dias_rango, n_registros))
    fecha = pd.to_datetime([fecha_inicio + timedelta(days=int(d)) for d in offsets])

    # 2) hora -> int 0-23 (más accidentes en horas pico)
    horas_pico = np.concatenate([np.arange(6, 9), np.arange(12, 14), np.arange(17, 20)])
    prob_horas = np.ones(24)
    prob_horas[horas_pico] = 3.0
    prob_horas = prob_horas / prob_horas.sum()
    hora = rng.choice(np.arange(24), size=n_registros, p=prob_horas)

    # 3) comuna -> categórica
    comuna = rng.choice(COMUNAS_MEDELLIN, size=n_registros, p=PESOS_COMUNAS)

    # 4) tipo_accidente -> categórica
    tipo_accidente = rng.choice(TIPOS_ACCIDENTE, size=n_registros, p=[0.45, 0.20, 0.15, 0.15, 0.05])

    # 5) clase_via -> categórica
    clase_via = rng.choice(CLASES_VIA, size=n_registros, p=[0.25, 0.30, 0.25, 0.12, 0.08])

    # 6) condicion_climatica -> categórica
    condicion_climatica = rng.choice(CONDICIONES_CLIMA, size=n_registros, p=[0.55, 0.25, 0.05, 0.15])

    # 7) num_vehiculos_involucrados -> int
    num_vehiculos = rng.poisson(lam=1.6, size=n_registros) + 1
    num_vehiculos = np.clip(num_vehiculos, 1, 8)

    # 8) velocidad_estimada_kmh -> float (mayor en autopistas/avenidas)
    base_vel = np.where(np.isin(clase_via, ["Autopista", "Avenida"]), 65, 38)
    velocidad_estimada = rng.normal(loc=base_vel, scale=12, size=n_registros)
    velocidad_estimada = np.round(np.clip(velocidad_estimada, 5, 130), 1)

    # 9) num_heridos -> int (más probable con mayor velocidad y clima adverso)
    factor_clima = np.where(np.isin(condicion_climatica, ["Lluvia", "Niebla"]), 1.4, 1.0)
    lam_heridos = np.clip((velocidad_estimada / 60) * factor_clima * 0.8, 0.05, 4)
    num_heridos = rng.poisson(lam=lam_heridos)

    # 10) num_muertos -> int (evento raro, ligado a velocidad alta)
    prob_muerte = np.clip((velocidad_estimada - 40) / 300, 0.001, 0.12)
    num_muertos = (rng.random(n_registros) < prob_muerte).astype(int)

    df = pd.DataFrame({
        "fecha": fecha,
        "hora": hora,
        "comuna": comuna,
        "tipo_accidente": tipo_accidente,
        "clase_via": clase_via,
        "condicion_climatica": condicion_climatica,
        "num_vehiculos_involucrados": num_vehiculos,
        "velocidad_estimada_kmh": velocidad_estimada,
        "num_heridos": num_heridos,
        "num_muertos": num_muertos,
    })

    return df.sort_values("fecha").reset_index(drop=True)


# ----------------------------------------------------------------------------
# SIDEBAR - CONTROLES DE GENERACIÓN Y FILTROS
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Generación de datos")

n_registros = st.sidebar.slider("Número de registros", min_value=100, max_value=2_000, value=500, step=50)
semilla = st.sidebar.number_input("Semilla aleatoria (reproducibilidad)", value=42, step=1)

if st.sidebar.button("🔄 Regenerar datos sintéticos"):
    generar_datos.clear()

df = generar_datos(n_registros, int(semilla))

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")

fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
rango_fechas = st.sidebar.date_input(
    "Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max
)
comunas_sel = st.sidebar.multiselect("Comuna", options=COMUNAS_MEDELLIN, default=COMUNAS_MEDELLIN)
tipo_sel = st.sidebar.multiselect("Tipo de accidente", options=TIPOS_ACCIDENTE, default=TIPOS_ACCIDENTE)
clima_sel = st.sidebar.multiselect("Condición climática", options=CONDICIONES_CLIMA, default=CONDICIONES_CLIMA)
via_sel = st.sidebar.multiselect("Clase de vía", options=CLASES_VIA, default=CLASES_VIA)

if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    f_ini, f_fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
else:
    f_ini, f_fin = fecha_min, fecha_max

df_f = df[
    (df["fecha"] >= f_ini) & (df["fecha"] <= f_fin) &
    (df["comuna"].isin(comunas_sel)) &
    (df["tipo_accidente"].isin(tipo_sel)) &
    (df["condicion_climatica"].isin(clima_sel)) &
    (df["clase_via"].isin(via_sel))
].copy()

st.sidebar.markdown(f"**Registros filtrados:** {len(df_f):,} / {len(df):,}")
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Cerrar sesión"):
    st.session_state["autenticado"] = False
    st.rerun()

COLS_NUM = ["hora", "num_vehiculos_involucrados", "velocidad_estimada_kmh", "num_heridos", "num_muertos"]
COLS_CAT = ["comuna", "tipo_accidente", "clase_via", "condicion_climatica"]

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🚦 Dashboard de Accidentes de Tránsito — Medellín (Datos Sintéticos)")
st.caption(
    "Todos los datos se generan de forma sintética dentro de la aplicación con fines "
    "académicos y demostrativos; no representan accidentes reales."
)

if df_f.empty:
    st.warning("No hay registros con los filtros actuales. Ajusta los filtros en la barra lateral.")
    st.stop()

tab_resumen, tab_cuant, tab_cual, tab_series, tab_viz = st.tabs(
    ["📋 Resumen", "📈 Cuantitativas", "🗂️ Cualitativas", "📅 Serie de Tiempo", "🎛️ Visualización Dinámica"]
)

# ----------------------------------------------------------------------------
# TAB 1: RESUMEN
# ----------------------------------------------------------------------------
with tab_resumen:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total accidentes", f"{len(df_f):,}")
    c2.metric("Heridos totales", f"{int(df_f['num_heridos'].sum()):,}")
    c3.metric("Muertos totales", f"{int(df_f['num_muertos'].sum()):,}")
    c4.metric("Velocidad promedio", f"{df_f['velocidad_estimada_kmh'].mean():.1f} km/h")
    tasa_letal = (df_f["num_muertos"] > 0).mean() * 100
    c5.metric("Accidentes con muertos", f"{tasa_letal:.1f}%")

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
        n_bins = st.slider("Número de bins", 5, 60, 20, key="bins_dist")
        fig_hist = px.histogram(df_f, x=var_num, nbins=n_bins, color="tipo_accidente",
                                 title=f"Histograma de {var_num}", opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(df_f, y=var_num, x="clase_via", color="clase_via", points="outliers",
                          title=f"Boxplot de {var_num} por clase de vía")
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
        var_cat_2 = st.selectbox("Variable 2", COLS_CAT, index=1, key="cross2")

    if var_cat_1 != var_cat_2:
        tabla_cruzada = pd.crosstab(df_f[var_cat_1], df_f[var_cat_2])
        st.dataframe(tabla_cruzada, use_container_width=True)
        fig_stack = px.bar(tabla_cruzada, barmode="stack", title=f"{var_cat_1} vs {var_cat_2}")
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Elige dos variables distintas para la tabla cruzada.")

# ----------------------------------------------------------------------------
# TAB 4: SERIE DE TIEMPO
# ----------------------------------------------------------------------------
with tab_series:
    st.markdown("### Evolución temporal de los accidentes")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        frecuencia = st.selectbox("Frecuencia de agregación", ["Diaria", "Semanal", "Mensual"], index=2)
    with col_s2:
        metrica_serie = st.selectbox(
            "Métrica a graficar",
            ["Número de accidentes", "Heridos", "Muertos", "Velocidad promedio"],
        )
    with col_s3:
        ventana_media_movil = st.slider("Ventana media móvil (periodos)", 0, 12, 3)

    freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}
    df_serie = df_f.set_index("fecha").sort_index()

    if metrica_serie == "Número de accidentes":
        serie = df_serie.resample(freq_map[frecuencia]).size().rename("valor")
    elif metrica_serie == "Heridos":
        serie = df_serie["num_heridos"].resample(freq_map[frecuencia]).sum().rename("valor")
    elif metrica_serie == "Muertos":
        serie = df_serie["num_muertos"].resample(freq_map[frecuencia]).sum().rename("valor")
    else:
        serie = df_serie["velocidad_estimada_kmh"].resample(freq_map[frecuencia]).mean().rename("valor")

    serie_df = serie.reset_index()

    fig_serie = go.Figure()
    fig_serie.add_trace(go.Scatter(x=serie_df["fecha"], y=serie_df["valor"], mode="lines+markers",
                                    name=metrica_serie, line=dict(color="#E45756")))

    if ventana_media_movil > 0:
        media_movil = serie_df["valor"].rolling(window=ventana_media_movil).mean()
        fig_serie.add_trace(go.Scatter(x=serie_df["fecha"], y=media_movil, mode="lines",
                                        name=f"Media móvil ({ventana_media_movil})",
                                        line=dict(color="#4C78A8", dash="dash")))

    activar_umbral_serie = st.checkbox("Activar umbral de referencia en la serie")
    if activar_umbral_serie and not serie_df["valor"].empty:
        min_v, max_v = float(serie_df["valor"].min()), float(serie_df["valor"].max())
        umbral_serie = st.slider("Valor del umbral", min_v, max_v, (min_v + max_v) / 2)
        fig_serie.add_hline(y=umbral_serie, line_dash="dot", line_color="green",
                            annotation_text=f"Umbral: {umbral_serie:.2f}")
        periodos_sobre = (serie_df["valor"] > umbral_serie).sum()
        st.caption(f"Periodos por encima del umbral: **{periodos_sobre}** de {len(serie_df)}")

    fig_serie.update_layout(title=f"{metrica_serie} — Agregación {frecuencia.lower()}",
                            xaxis_title="Fecha", yaxis_title=metrica_serie,
                            template="plotly_white", height=550)
    st.plotly_chart(fig_serie, use_container_width=True)

    st.markdown("#### Accidentes por hora del día")
    conteo_hora = df_f["hora"].value_counts().sort_index().reset_index()
    conteo_hora.columns = ["hora", "conteo"]
    fig_hora = px.bar(conteo_hora, x="hora", y="conteo", title="Distribución de accidentes por hora del día",
                       color="conteo", color_continuous_scale="Reds")
    st.plotly_chart(fig_hora, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 5: VISUALIZACIÓN DINÁMICA E INTERACTIVA
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
        eje_x = st.selectbox("Variable eje X", todas_cols, index=todas_cols.index("fecha"))
    with col3:
        eje_y_opciones = ["(ninguna)"] + todas_cols
        idx_y = eje_y_opciones.index("velocidad_estimada_kmh") if "velocidad_estimada_kmh" in eje_y_opciones else 0
        eje_y = st.selectbox("Variable eje Y", eje_y_opciones, index=idx_y)

    col4, col5, col6 = st.columns(3)
    with col4:
        color_por = st.selectbox("Colorear por", ["(ninguna)"] + COLS_CAT)
    with col5:
        paleta = st.selectbox("Paleta de colores", ["Plotly", "Vivid", "Bold", "Pastel", "Set2", "D3"])
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
    st.download_button("⬇️ Descargar CSV filtrado", data=csv, file_name="accidentes_medellin_sinteticos.csv",
                       mime="text/csv")
