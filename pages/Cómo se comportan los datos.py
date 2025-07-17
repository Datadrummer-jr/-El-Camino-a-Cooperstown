<<<<<<< HEAD
import streamlit as st
from PIL import Image
import pandas as pd
import json
import plotly.express as px

logo = Image.open("logo.jpg")

title = "¿Cómo se comportan los datos?"

st.set_page_config(
    page_title= title,
    page_icon=logo,
    layout="wide")

# Función para cargar los datos del Salón de la Fama 
@st.cache_data
def load_hof_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que esté en la misma carpeta.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: No se pudo decodificar el JSON de '{file_path}'. Verifica su formato.")
        st.stop()
    
    records = []
    for player_name, player_data in data.items():
        record = {"Name": player_name}
        record.update(player_data)
        records.append(record)
    
    df = pd.DataFrame(records)
    
    numeric_cols = [
        "% of Ballots", "induction", "first_game", "last_game", 
        "years_of_waiting_to_enter", "years_of_experience", "war", 
        "g_bat", "h", "hr", "ab", "ba", "rbi", "obp", "ops", "slg",
        "l", "w", "era", "war_p", "g", "bb", "gf", "w_l", "ip"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# Función para cargar los datos de países 
@st.cache_data
def load_country_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning(f"Advertencia: El archivo de países '{file_path}' no se encontró. El análisis por nacionalidad no estará disponible.")
        return pd.DataFrame(columns=["name", "country"]) 
    except json.JSONDecodeError:
        st.error(f"Error: No se pudo decodificar el JSON de países de '{file_path}'. Verifica su formato.")
        st.stop()
    
    # Convertir el JSON a una lista de diccionarios para el DataFrame
    country_records = [value for key, value in data.items()]
    df_countries = pd.DataFrame(country_records)
    return df_countries

# Cargar los datos
hof_file_path = 'hof.json'
country_file_path = 'nationality.json'

df_hof = load_hof_data(hof_file_path)
df_countries = load_country_data(country_file_path)

# Fusionar los DataFrames
df = pd.merge(df_hof, df_countries, left_on='Name', right_on='name', how='left')
df.drop(columns=['name'], inplace=True, errors='ignore')

st.title("⚾ Dashboard Extendido del Salón de la Fama del Béisbol (Plotly)")
st.markdown("Explora en profundidad las estadísticas de los jugadores y managers con **más de 10 gráficos interactivos**, incluyendo **análisis por nacionalidad**.")

# Sidebar para filtros
st.sidebar.header("Filtros y Opciones")

player_names = df["Name"].unique().tolist()
selected_players = st.sidebar.multiselect(
    "Selecciona Jugadores/Managers:",
    player_names,
    default=player_names 
)

inducted_as_options = df["inducted_as"].unique().tolist()
selected_inducted_as = st.sidebar.multiselect(
    "Filtrar por Inducción como:",
    inducted_as_options,
    default=inducted_as_options
)

min_induction_year = int(df["induction"].min()) if not df["induction"].isnull().all() else 1900
max_induction_year = int(df["induction"].max()) if not df["induction"].isnull().all() else 2025
induction_year_range = st.sidebar.slider(
    "Rango de Año de Inducción:",
    min_value=min_induction_year,
    max_value=max_induction_year,
    value=(min_induction_year, max_induction_year)
)

# Aplicar filtros
filtered_df = df[
    (df["Name"].isin(selected_players)) &
    (df["inducted_as"].isin(selected_inducted_as)) &
    (df["induction"] >= induction_year_range[0]) & 
    (df["induction"] <= induction_year_range[1])
]

# Mostrar los datos filtrados 
st.header("Datos Filtrados")
if not filtered_df.empty:
    st.dataframe(filtered_df.drop(columns=['link'], errors='ignore')) 
    st.markdown("---")
else:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
    st.stop() 

# Sección de Visualizaciones
st.header("Visualizaciones Interactivas")

# Porcentaje de Balotas al Ser Inducido
st.subheader("1. Porcentaje de Balotas al Ser Inducido")
players_for_ballots = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["% of Ballots"].notna())
].sort_values(by="% of Ballots", ascending=False).head(20) 
if not players_for_ballots.empty:
    fig1 = px.bar(
        players_for_ballots, x="Name", y="% of Ballots", 
        title="Top 20 de Porcentaje de Balotas para la Inducción",
        labels={"Name": "Jugador", "% of Ballots": "% de Balotas"},
        hover_data={"% of Ballots": ":.1f"}
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No hay datos de porcentaje de boletas.")

st.markdown("---")

# Gráfico 2: WAR (Victorias por Encima del Reemplazo)
st.subheader("2. WAR (Victorias por Encima del Reemplazo)")
war_df = filtered_df[filtered_df["war"].notna()].sort_values(by="war", ascending=False).head(20)
if not war_df.empty:
    fig2 = px.bar(
        war_df, x="Name", y="war", 
        title="Top 20 de WAR (Victorias por Encima del Reemplazo)",
        labels={"Name": "Jugador", "war": "WAR"},
        hover_data={"war": ":.1f", "inducted_as": True}
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No hay datos de WAR para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 3: Home Runs (HR) para bateadores
st.subheader("3. Home Runs (HR) para Jugadores")
batters_hr_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["hr"].notna())
].sort_values(by="hr", ascending=False).head(20)
if not batters_hr_df.empty:
    fig3 = px.bar(
        batters_hr_df, x="Name", y="hr", 
        title="Top 20 de Home Runs",
        labels={"Name": "Jugador", "hr": "Home Runs"},
        hover_data={"hr": True, "years_of_experience": True}
    )
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No hay datos de Home Runs para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 4: Promedio de Bateo (BA) para bateadores
st.subheader("4. Promedio de Bateo (BA) para Jugadores")
batters_ba_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["ba"].notna())
].sort_values(by="ba", ascending=False).head(20)
if not batters_ba_df.empty:
    fig4 = px.bar(
        batters_ba_df, x="Name", y="ba", 
        title="Top 20 de Promedio de Bateo",
        labels={"Name": "Jugador", "ba": "Promedio de Bateo"},
        hover_data={"ba": ":.3f", "h": True, "ab": True}
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No hay datos de Promedio de Bateo para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 5: ERA para Lanzadores
st.subheader("5. Promedio de Carreras Limpias (ERA) para Lanzadores")
pitchers_era_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["era"].notna())
].sort_values(by="era").head(20) 
if not pitchers_era_df.empty:
    fig5 = px.bar(
        pitchers_era_df, x="Name", y="era", 
        title="Top 20 de ERA para Lanzadores (más bajos son mejores)",
        labels={"Name": "Lanzador", "era": "ERA"},
        hover_data={"era": ":.2f", "w": True, "l": True, "ip": ":.1f"}
    )
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("No hay datos de ERA para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 6: Victorias (W) para Lanzadores
st.subheader("6. Victorias (W) para Lanzadores")
pitchers_w_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["w"].notna())
].sort_values(by="w", ascending=False).head(20)
if not pitchers_w_df.empty:
    fig6 = px.bar(
        pitchers_w_df, x="Name", y="w", 
        title="Top 20 de Victorias para Lanzadores",
        labels={"Name": "Lanzador", "w": "Victorias"},
        hover_data={"w": True, "l": True, "w_l": ":.3f"}
    )
    fig6.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("No hay datos de Victorias para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 7: Relación WAR vs. HR (Scatter plot)
st.subheader("7. Relación entre WAR y Home Runs")

# Filtrar WAR para asegurar que no sean negativos o nulos antes de usar en 'size'
war_hr_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["war"].notna()) & 
    (filtered_df["war"] >= 0) & 
    (filtered_df["hr"].notna())
]
if not war_hr_df.empty:
    fig7 = px.scatter(
        war_hr_df, x="hr", y="war", color="Name", size="war",
        title="WAR vs. Home Runs",
        labels={"hr": "Home Runs", "war": "WAR"},
        hover_data={"Name": True, "hr": True, "war": ":.1f", "years_of_experience": True}
    )
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("No hay datos suficientes de WAR (positivos) y Home Runs para este gráfico.")

st.markdown("---")

# Gráfico 8: Relación ERA vs. W (Scatter plot para lanzadores)
st.subheader("8. Relación entre ERA y Victorias para Lanzadores")
era_w_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["era"].notna()) & 
    (filtered_df["w"].notna())
]
if not era_w_df.empty:
    fig8 = px.scatter(
        era_w_df, x="w", y="era", color="Name", size="w", 
        title="ERA vs. Victorias para Lanzadores",
        labels={"w": "Victorias (W)", "era": "ERA"},
        hover_data={"Name": True, "w": True, "era": ":.2f", "l": True, "ip": ":.1f"}
    )
    # Invertir el eje Y para que los ERA más bajos (mejores) estén arriba
    fig8.update_yaxes(autorange="reversed")
    st.plotly_chart(fig8, use_container_width=True)
else:
    st.info("No hay datos suficientes de ERA y Victorias para este gráfico.")

st.markdown("---")

# Años de Experiencia
st.subheader("9. Distribución de Años de Experiencia")
exp_df = filtered_df[
    (filtered_df["years_of_experience"].notna()) & 
    (filtered_df["inducted_as"] == "Player")
]
if not exp_df.empty:
    fig9 = px.histogram(
        exp_df, x="years_of_experience",
        title="Distribución de Años de Experiencia de Jugadores",
        labels={"years_of_experience": "Años de Experiencia"},
        nbins=10, 
        hover_data={"Name": True}
    )
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("No hay datos de años de experiencia para mostrar.")

st.markdown("---")

# Promedio de Bateo (BA) vs. Promedio de Embasado (OBP)
st.subheader("10. Promedio de Bateo (BA) vs. Promedio de Embasado (OBP)")

# Filtrar OPS para asegurar que no sean negativos o nulos antes de usar en 'size'
ba_obp_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["ba"].notna()) & 
    (filtered_df["obp"].notna()) &
    (filtered_df["ops"].notna()) &
    (filtered_df["ops"] >= 0) 
]
if not ba_obp_df.empty:
    fig10 = px.scatter(
        ba_obp_df, x="ba", y="obp", color="Name", size="ops",
        title="BA vs. OBP para Jugadores",
        labels={"ba": "Promedio de Bateo (BA)", "obp": "Promedio de Embasado (OBP)"},
        hover_data={"Name": True, "ba": ":.3f", "obp": ":.3f", "ops": ":.3f"}
    )
    st.plotly_chart(fig10, use_container_width=True)
else:
    st.info("No hay datos suficientes de BA, OBP o OPS (positivos) para este gráfico.")

st.markdown("---")

# Distribución por Nacionalidad
st.subheader("11. Distribución de Miembros por Nacionalidad")
country_counts = filtered_df['country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

if not country_counts.empty:
    fig11 = px.bar(
        country_counts, 
        x='Country', 
        y='Count', 
        title='Número de Miembros del Salón de la Fama por País',
        labels={'Country': 'País', 'Count': 'Número de Miembros'},
        color='Country'
    )
    fig11.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig11, use_container_width=True)
else:
    st.info("No hay datos de nacionalidad para mostrar con los filtros actuales. Asegúrate de que el archivo 'deportistas_paises.json' exista y los nombres de los jugadores coincidan.")

st.markdown("---")

# Información detallada del jugador seleccionado
if len(selected_players) == 1:
    st.header(f"Detalles de {selected_players[0]}")
    player_detail = filtered_df[filtered_df["Name"] == selected_players[0]].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información General")
        st.write(f"**Inducido como:** {player_detail.get('inducted_as', 'N/A')}")
        st.write(f"**Año de Inducción:** {int(player_detail['induction']) if pd.notna(player_detail['induction']) else 'N/A'}")
        st.write(f"**Primer Juego:** {int(player_detail['first_game']) if pd.notna(player_detail['first_game']) else 'N/A'}")
        st.write(f"**Último Juego:** {int(player_detail['last_game']) if pd.notna(player_detail['last_game']) else 'N/A'}")
        st.write(f"**Años de Espera para Entrar:** {int(player_detail['years_of_waiting_to_enter']) if pd.notna(player_detail['years_of_waiting_to_enter']) else 'N/A'}")
        st.write(f"**Años de Experiencia:** {int(player_detail['years_of_experience']) if pd.notna(player_detail['years_of_experience']) else 'N/A'}")
        if pd.notna(player_detail.get('% of Ballots')):
            st.write(f"**% de Balotas:** {player_detail['% of Ballots']:.1f}%")
        if player_detail.get('link'):
            st.markdown(f"**Más info:** [Baseball-Reference]({player_detail['link']})")
        if pd.notna(player_detail.get('country')):
            st.write(f"**Nacionalidad:** {player_detail['country']}")


    with col2:
        st.subheader("Estadísticas de Bateo")
        if pd.notna(player_detail.get('g_bat')):
            st.write(f"**Juegos (Bat):** {int(player_detail['g_bat'])}")
        if pd.notna(player_detail.get('h')):
            st.write(f"**Hits (H):** {int(player_detail['h'])}")
        if pd.notna(player_detail.get('hr')):
            st.write(f"**Home Runs (HR):** {int(player_detail['hr'])}")
        if pd.notna(player_detail.get('rbi')):
            st.write(f"**Carreras Impulsadas (RBI):** {int(player_detail['rbi'])}")
        if pd.notna(player_detail.get('ab')):
            st.write(f"**Turnos al Bate (AB):** {int(player_detail['ab'])}")
        if pd.notna(player_detail.get('ba')):
            st.write(f"**Promedio de Bateo (BA):** {player_detail['ba']:.3f}")
        if pd.notna(player_detail.get('obp')):
            st.write(f"**Promedio de Embasado (OBP):** {player_detail['obp']:.3f}")
        if pd.notna(player_detail.get('slg')):
            st.write(f"**Slugging (SLG):** {player_detail['slg']:.3f}")
        if pd.notna(player_detail.get('ops')):
            st.write(f"**OPS:** {player_detail['ops']:.3f}")
        if pd.notna(player_detail.get('war')):
            st.write(f"**WAR (Total):** {player_detail['war']:.1f}")

    # Estadísticas de Pitcheo (si aplica)
    if pd.notna(player_detail.get('w')) or pd.notna(player_detail.get('era')):
        st.subheader("Estadísticas de Pitcheo")
        if pd.notna(player_detail.get('w')):
            st.write(f"**Victorias (W):** {int(player_detail['w'])}")
        if pd.notna(player_detail.get('l')):
            st.write(f"**Derrotas (L):** {int(player_detail['l'])}")
        if pd.notna(player_detail.get('era')):
            st.write(f"**ERA:** {player_detail['era']:.2f}")
        if pd.notna(player_detail.get('war_p')):
            st.write(f"**WAR (Pitcheo):** {player_detail['war_p']:.2f}")
        if pd.notna(player_detail.get('g')):
            st.write(f"**Juegos (Pitch):** {int(player_detail['g'])}")
        if pd.notna(player_detail.get('ip')):
            st.write(f"**Entradas Lanzadas (IP):** {player_detail['ip']:.1f}")
        if pd.notna(player_detail.get('bb')):
            st.write(f"**Bases por Bola (BB):** {int(player_detail['bb'])}")
        if pd.notna(player_detail.get('w_l')):
            st.write(f"**Porcentaje de Victorias-Derrotas (W-L%):** {player_detail['w_l']:.3f}")

elif len(selected_players) > 1:
    st.info("Selecciona un solo jugador en el filtro de la barra lateral para ver sus detalles completos.")

=======
import streamlit as st
from PIL import Image
import pandas as pd
import json
import plotly.express as px

logo = Image.open("logo.jpg")

title = "¿Cómo se comportan los datos?"

st.set_page_config(
    page_title= title,
    page_icon=logo,
    layout="wide")

# Función para cargar los datos del Salón de la Fama 
@st.cache_data
def load_hof_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que esté en la misma carpeta.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: No se pudo decodificar el JSON de '{file_path}'. Verifica su formato.")
        st.stop()
    
    records = []
    for player_name, player_data in data.items():
        record = {"Name": player_name}
        record.update(player_data)
        records.append(record)
    
    df = pd.DataFrame(records)
    
    numeric_cols = [
        "% of Ballots", "induction", "first_game", "last_game", 
        "years_of_waiting_to_enter", "years_of_experience", "war", 
        "g_bat", "h", "hr", "ab", "ba", "rbi", "obp", "ops", "slg",
        "l", "w", "era", "war_p", "g", "bb", "gf", "w_l", "ip"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# Función para cargar los datos de países 
@st.cache_data
def load_country_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.warning(f"Advertencia: El archivo de países '{file_path}' no se encontró. El análisis por nacionalidad no estará disponible.")
        return pd.DataFrame(columns=["name", "country"]) 
    except json.JSONDecodeError:
        st.error(f"Error: No se pudo decodificar el JSON de países de '{file_path}'. Verifica su formato.")
        st.stop()
    
    # Convertir el JSON a una lista de diccionarios para el DataFrame
    country_records = [value for key, value in data.items()]
    df_countries = pd.DataFrame(country_records)
    return df_countries

# Cargar los datos
hof_file_path = 'hof.json'
country_file_path = 'nationality.json'

df_hof = load_hof_data(hof_file_path)
df_countries = load_country_data(country_file_path)

# Fusionar los DataFrames
df = pd.merge(df_hof, df_countries, left_on='Name', right_on='name', how='left')
df.drop(columns=['name'], inplace=True, errors='ignore')

st.title("⚾ Dashboard Extendido del Salón de la Fama del Béisbol (Plotly)")
st.markdown("Explora en profundidad las estadísticas de los jugadores y managers con **más de 10 gráficos interactivos**, incluyendo **análisis por nacionalidad**.")

# Sidebar para filtros
st.sidebar.header("Filtros y Opciones")

player_names = df["Name"].unique().tolist()
selected_players = st.sidebar.multiselect(
    "Selecciona Jugadores/Managers:",
    player_names,
    default=player_names 
)

inducted_as_options = df["inducted_as"].unique().tolist()
selected_inducted_as = st.sidebar.multiselect(
    "Filtrar por Inducción como:",
    inducted_as_options,
    default=inducted_as_options
)

min_induction_year = int(df["induction"].min()) if not df["induction"].isnull().all() else 1900
max_induction_year = int(df["induction"].max()) if not df["induction"].isnull().all() else 2025
induction_year_range = st.sidebar.slider(
    "Rango de Año de Inducción:",
    min_value=min_induction_year,
    max_value=max_induction_year,
    value=(min_induction_year, max_induction_year)
)

# Aplicar filtros
filtered_df = df[
    (df["Name"].isin(selected_players)) &
    (df["inducted_as"].isin(selected_inducted_as)) &
    (df["induction"] >= induction_year_range[0]) & 
    (df["induction"] <= induction_year_range[1])
]

# Mostrar los datos filtrados 
st.header("Datos Filtrados")
if not filtered_df.empty:
    st.dataframe(filtered_df.drop(columns=['link'], errors='ignore')) 
    st.markdown("---")
else:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
    st.stop() 

# Sección de Visualizaciones
st.header("Visualizaciones Interactivas")

# Porcentaje de Balotas al Ser Inducido
st.subheader("1. Porcentaje de Balotas al Ser Inducido")
players_for_ballots = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["% of Ballots"].notna())
].sort_values(by="% of Ballots", ascending=False).head(20) 
if not players_for_ballots.empty:
    fig1 = px.bar(
        players_for_ballots, x="Name", y="% of Ballots", 
        title="Top 20 de Porcentaje de Balotas para la Inducción",
        labels={"Name": "Jugador", "% of Ballots": "% de Balotas"},
        hover_data={"% of Ballots": ":.1f"}
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No hay datos de porcentaje de boletas.")

st.markdown("---")

# Gráfico 2: WAR (Victorias por Encima del Reemplazo)
st.subheader("2. WAR (Victorias por Encima del Reemplazo)")
war_df = filtered_df[filtered_df["war"].notna()].sort_values(by="war", ascending=False).head(20)
if not war_df.empty:
    fig2 = px.bar(
        war_df, x="Name", y="war", 
        title="Top 20 de WAR (Victorias por Encima del Reemplazo)",
        labels={"Name": "Jugador", "war": "WAR"},
        hover_data={"war": ":.1f", "inducted_as": True}
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No hay datos de WAR para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 3: Home Runs (HR) para bateadores
st.subheader("3. Home Runs (HR) para Jugadores")
batters_hr_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["hr"].notna())
].sort_values(by="hr", ascending=False).head(20)
if not batters_hr_df.empty:
    fig3 = px.bar(
        batters_hr_df, x="Name", y="hr", 
        title="Top 20 de Home Runs",
        labels={"Name": "Jugador", "hr": "Home Runs"},
        hover_data={"hr": True, "years_of_experience": True}
    )
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No hay datos de Home Runs para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 4: Promedio de Bateo (BA) para bateadores
st.subheader("4. Promedio de Bateo (BA) para Jugadores")
batters_ba_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["ba"].notna())
].sort_values(by="ba", ascending=False).head(20)
if not batters_ba_df.empty:
    fig4 = px.bar(
        batters_ba_df, x="Name", y="ba", 
        title="Top 20 de Promedio de Bateo",
        labels={"Name": "Jugador", "ba": "Promedio de Bateo"},
        hover_data={"ba": ":.3f", "h": True, "ab": True}
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No hay datos de Promedio de Bateo para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 5: ERA para Lanzadores
st.subheader("5. Promedio de Carreras Limpias (ERA) para Lanzadores")
pitchers_era_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["era"].notna())
].sort_values(by="era").head(20) 
if not pitchers_era_df.empty:
    fig5 = px.bar(
        pitchers_era_df, x="Name", y="era", 
        title="Top 20 de ERA para Lanzadores (más bajos son mejores)",
        labels={"Name": "Lanzador", "era": "ERA"},
        hover_data={"era": ":.2f", "w": True, "l": True, "ip": ":.1f"}
    )
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("No hay datos de ERA para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 6: Victorias (W) para Lanzadores
st.subheader("6. Victorias (W) para Lanzadores")
pitchers_w_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["w"].notna())
].sort_values(by="w", ascending=False).head(20)
if not pitchers_w_df.empty:
    fig6 = px.bar(
        pitchers_w_df, x="Name", y="w", 
        title="Top 20 de Victorias para Lanzadores",
        labels={"Name": "Lanzador", "w": "Victorias"},
        hover_data={"w": True, "l": True, "w_l": ":.3f"}
    )
    fig6.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("No hay datos de Victorias para mostrar con los filtros actuales.")

st.markdown("---")

# Gráfico 7: Relación WAR vs. HR (Scatter plot)
st.subheader("7. Relación entre WAR y Home Runs")

# Filtrar WAR para asegurar que no sean negativos o nulos antes de usar en 'size'
war_hr_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["war"].notna()) & 
    (filtered_df["war"] >= 0) & 
    (filtered_df["hr"].notna())
]
if not war_hr_df.empty:
    fig7 = px.scatter(
        war_hr_df, x="hr", y="war", color="Name", size="war",
        title="WAR vs. Home Runs",
        labels={"hr": "Home Runs", "war": "WAR"},
        hover_data={"Name": True, "hr": True, "war": ":.1f", "years_of_experience": True}
    )
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("No hay datos suficientes de WAR (positivos) y Home Runs para este gráfico.")

st.markdown("---")

# Gráfico 8: Relación ERA vs. W (Scatter plot para lanzadores)
st.subheader("8. Relación entre ERA y Victorias para Lanzadores")
era_w_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["era"].notna()) & 
    (filtered_df["w"].notna())
]
if not era_w_df.empty:
    fig8 = px.scatter(
        era_w_df, x="w", y="era", color="Name", size="w", 
        title="ERA vs. Victorias para Lanzadores",
        labels={"w": "Victorias (W)", "era": "ERA"},
        hover_data={"Name": True, "w": True, "era": ":.2f", "l": True, "ip": ":.1f"}
    )
    # Invertir el eje Y para que los ERA más bajos (mejores) estén arriba
    fig8.update_yaxes(autorange="reversed")
    st.plotly_chart(fig8, use_container_width=True)
else:
    st.info("No hay datos suficientes de ERA y Victorias para este gráfico.")

st.markdown("---")

# Años de Experiencia
st.subheader("9. Distribución de Años de Experiencia")
exp_df = filtered_df[
    (filtered_df["years_of_experience"].notna()) & 
    (filtered_df["inducted_as"] == "Player")
]
if not exp_df.empty:
    fig9 = px.histogram(
        exp_df, x="years_of_experience",
        title="Distribución de Años de Experiencia de Jugadores",
        labels={"years_of_experience": "Años de Experiencia"},
        nbins=10, 
        hover_data={"Name": True}
    )
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("No hay datos de años de experiencia para mostrar.")

st.markdown("---")

# Promedio de Bateo (BA) vs. Promedio de Embasado (OBP)
st.subheader("10. Promedio de Bateo (BA) vs. Promedio de Embasado (OBP)")

# Filtrar OPS para asegurar que no sean negativos o nulos antes de usar en 'size'
ba_obp_df = filtered_df[
    (filtered_df["inducted_as"] == "Player") & 
    (filtered_df["ba"].notna()) & 
    (filtered_df["obp"].notna()) &
    (filtered_df["ops"].notna()) &
    (filtered_df["ops"] >= 0) 
]
if not ba_obp_df.empty:
    fig10 = px.scatter(
        ba_obp_df, x="ba", y="obp", color="Name", size="ops",
        title="BA vs. OBP para Jugadores",
        labels={"ba": "Promedio de Bateo (BA)", "obp": "Promedio de Embasado (OBP)"},
        hover_data={"Name": True, "ba": ":.3f", "obp": ":.3f", "ops": ":.3f"}
    )
    st.plotly_chart(fig10, use_container_width=True)
else:
    st.info("No hay datos suficientes de BA, OBP o OPS (positivos) para este gráfico.")

st.markdown("---")

# Distribución por Nacionalidad
st.subheader("11. Distribución de Miembros por Nacionalidad")
country_counts = filtered_df['country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

if not country_counts.empty:
    fig11 = px.bar(
        country_counts, 
        x='Country', 
        y='Count', 
        title='Número de Miembros del Salón de la Fama por País',
        labels={'Country': 'País', 'Count': 'Número de Miembros'},
        color='Country'
    )
    fig11.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig11, use_container_width=True)
else:
    st.info("No hay datos de nacionalidad para mostrar con los filtros actuales. Asegúrate de que el archivo 'deportistas_paises.json' exista y los nombres de los jugadores coincidan.")

st.markdown("---")

# Información detallada del jugador seleccionado
if len(selected_players) == 1:
    st.header(f"Detalles de {selected_players[0]}")
    player_detail = filtered_df[filtered_df["Name"] == selected_players[0]].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información General")
        st.write(f"**Inducido como:** {player_detail.get('inducted_as', 'N/A')}")
        st.write(f"**Año de Inducción:** {int(player_detail['induction']) if pd.notna(player_detail['induction']) else 'N/A'}")
        st.write(f"**Primer Juego:** {int(player_detail['first_game']) if pd.notna(player_detail['first_game']) else 'N/A'}")
        st.write(f"**Último Juego:** {int(player_detail['last_game']) if pd.notna(player_detail['last_game']) else 'N/A'}")
        st.write(f"**Años de Espera para Entrar:** {int(player_detail['years_of_waiting_to_enter']) if pd.notna(player_detail['years_of_waiting_to_enter']) else 'N/A'}")
        st.write(f"**Años de Experiencia:** {int(player_detail['years_of_experience']) if pd.notna(player_detail['years_of_experience']) else 'N/A'}")
        if pd.notna(player_detail.get('% of Ballots')):
            st.write(f"**% de Balotas:** {player_detail['% of Ballots']:.1f}%")
        if player_detail.get('link'):
            st.markdown(f"**Más info:** [Baseball-Reference]({player_detail['link']})")
        if pd.notna(player_detail.get('country')):
            st.write(f"**Nacionalidad:** {player_detail['country']}")


    with col2:
        st.subheader("Estadísticas de Bateo")
        if pd.notna(player_detail.get('g_bat')):
            st.write(f"**Juegos (Bat):** {int(player_detail['g_bat'])}")
        if pd.notna(player_detail.get('h')):
            st.write(f"**Hits (H):** {int(player_detail['h'])}")
        if pd.notna(player_detail.get('hr')):
            st.write(f"**Home Runs (HR):** {int(player_detail['hr'])}")
        if pd.notna(player_detail.get('rbi')):
            st.write(f"**Carreras Impulsadas (RBI):** {int(player_detail['rbi'])}")
        if pd.notna(player_detail.get('ab')):
            st.write(f"**Turnos al Bate (AB):** {int(player_detail['ab'])}")
        if pd.notna(player_detail.get('ba')):
            st.write(f"**Promedio de Bateo (BA):** {player_detail['ba']:.3f}")
        if pd.notna(player_detail.get('obp')):
            st.write(f"**Promedio de Embasado (OBP):** {player_detail['obp']:.3f}")
        if pd.notna(player_detail.get('slg')):
            st.write(f"**Slugging (SLG):** {player_detail['slg']:.3f}")
        if pd.notna(player_detail.get('ops')):
            st.write(f"**OPS:** {player_detail['ops']:.3f}")
        if pd.notna(player_detail.get('war')):
            st.write(f"**WAR (Total):** {player_detail['war']:.1f}")

    # Estadísticas de Pitcheo (si aplica)
    if pd.notna(player_detail.get('w')) or pd.notna(player_detail.get('era')):
        st.subheader("Estadísticas de Pitcheo")
        if pd.notna(player_detail.get('w')):
            st.write(f"**Victorias (W):** {int(player_detail['w'])}")
        if pd.notna(player_detail.get('l')):
            st.write(f"**Derrotas (L):** {int(player_detail['l'])}")
        if pd.notna(player_detail.get('era')):
            st.write(f"**ERA:** {player_detail['era']:.2f}")
        if pd.notna(player_detail.get('war_p')):
            st.write(f"**WAR (Pitcheo):** {player_detail['war_p']:.2f}")
        if pd.notna(player_detail.get('g')):
            st.write(f"**Juegos (Pitch):** {int(player_detail['g'])}")
        if pd.notna(player_detail.get('ip')):
            st.write(f"**Entradas Lanzadas (IP):** {player_detail['ip']:.1f}")
        if pd.notna(player_detail.get('bb')):
            st.write(f"**Bases por Bola (BB):** {int(player_detail['bb'])}")
        if pd.notna(player_detail.get('w_l')):
            st.write(f"**Porcentaje de Victorias-Derrotas (W-L%):** {player_detail['w_l']:.3f}")

elif len(selected_players) > 1:
    st.info("Selecciona un solo jugador en el filtro de la barra lateral para ver sus detalles completos.")

>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
