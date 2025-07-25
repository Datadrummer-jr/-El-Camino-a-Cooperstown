import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_library import my_library as ml

logo = Image.open("imagenes/logo.jpg")
salón = Image.open('imagenes/Salón.jpg')
placas = Image.open('imagenes/Placas.jpg')
Babe_Ruth = Image.open('imagenes/Babe_Ruth_1922.jpeg')
Mariano_Rivera = Image.open('imagenes/Mariano_Rivera.jpeg')

title="Camino a Cooperstown"

st.set_page_config(
    page_title=title,
    page_icon=logo,
    layout="wide")

df = pd.read_json("hof.json")

pd.set_option("display.max_rows", None)

df_total = df.T

df_players = df.T[~df.T["inducted_as"].isin(["Manager", "Pioneer/Executive", "Umpire"])]

df_managers =  df.T[~df.T["inducted_as"].isin(["Player", "Pioneer/Executive", "Umpire"])]

df_Pioneer_Executive =  df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Umpire"])]

df_umpire = df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Pioneer/Executive"])]

df_batting = df_players.copy()

categorias = {
    "Jugadores": df_players,
    "Managers": df_managers,
    "Ejecutivos": df_Pioneer_Executive,
    "Árbitros": df_umpire,
    "todos" : df_total
}

df_players_copy = df_players.copy()

df_players_copy['Total_de_Juegos'] = df_players_copy[['g','g_bat']].sum(axis=1,skipna=True)
df_players_copy['Total_de_WAR'] = df_players_copy[['war','war_p']].sum(axis=1,skipna=True)

batt = df_players_copy.dropna(subset=['% of Ballots','years_of_experience'])
df_batt = batt[batt['g'].isna()]

df_batting = df_players_copy[df_players_copy["g"].isna()]
df_ambos = df_players_copy.dropna(subset=['g','g_bat'])

ind_batt = df_batt[['years_of_experience','g_bat','war', 'h','hr','ab','ba','ops','obp','rbi','slg']]
dep_batt = df_batt['% of Ballots']
    
Gx = PolynomialFeatures(degree=2)
bx = Gx.fit_transform(ind_batt)
model_batting = LinearRegression()
model_batting.fit(bx,dep_batt)

ind_test = df_ambos[[ 'years_of_experience','g_bat','war', 'h','hr','ab','ba','ops','obp','rbi', 'slg']]

df_ambos['% of Ballots as Batting Predicho'] = model_batting.predict(Gx.fit_transform(df_ambos[ind_test.columns]))

df_batting = pd.concat([df_ambos[df_ambos['% of Ballots as Batting Predicho'] >= 75], df_batting],axis=0)
df_pitching = df_ambos[df_ambos['% of Ballots as Batting Predicho'] < 75]

columnas_pit = ["era", "war_p","g" ,"l","w",'ip','bb','w_l','years_of_experience','gf',"% of Ballots"]
df_pitch = df_pitching.dropna(subset=columnas_pit)

columnas_bat = ['years_of_experience', "% of Ballots", "war", "g_bat", "h", "hr", "ba","ab" ,"rbi","obp","ops",'slg' ]
df_batt = df_batting.dropna(subset=columnas_bat)

# Variables Independientes para los pitcher

era_pit = df_pitch['era'].to_list()
war_pit =  df_pitch["Total_de_WAR"].to_list()
g_pit =  df_pitch["Total_de_Juegos"].to_list()
l_pit =  df_pitch["l"].to_list()
w_pit =  df_pitch["w"].to_list()
ip_pit =  df_pitch['ip'].to_list()
bb_pit =  df_pitch['bb'].to_list()
w_l_pit =  df_pitch['w_l'].to_list()
gf_pit =  df_pitch['gf'].to_list()
experience_pit =  df_pitch['years_of_experience'].to_list()

#Variables independientes para los bateadores

war_bat = df_batt["war"].to_list()
g_bat = df_batt["g_bat"].to_list()
h_bat = df_batt["h"].to_list()
hr_bat = df_batt["hr"].to_list()
ba_bat = df_batt["ba"].to_list()
ab_bat = df_batt["ab"].to_list()
rbi_bat = df_batt["rbi"].to_list()
obp_bat = df_batt["obp"].to_list()
ops_bat = df_batt["ops"].to_list()
slg_bat = df_batt["slg"].to_list()
experience_bat = df_batt["years_of_experience"].to_list()

#variables dependiente para los pitcher

porcent_pit =  df_pitch["% of Ballots"].to_list()
demora_pit =  df_pitch["years_of_waiting_to_enter"].to_list()

# Variables dependiente para los bateadores

porcent_bat = df_batt["% of Ballots"].to_list()
demora_bat = df_batt["years_of_waiting_to_enter"].to_list()

# Entrenamiento del modelo de Regresión Lineal Múltiple para predecir el por ciento de las boletas para los batting

X = np.array([experience_bat,g_bat, war_bat,h_bat,hr_bat,ab_bat,ba_bat,rbi_bat,ops_bat,obp_bat,slg_bat]).T

y = np.array(porcent_bat)

Poly_bat = PolynomialFeatures(degree=4)

Px = Poly_bat.fit_transform(X)

model_bat = LinearRegression()
model_bat.fit(Px,y)

W = np.array([experience_pit,g_pit, gf_pit,war_pit,era_pit,l_pit,bb_pit,w_pit,w_l_pit,ip_pit]).T

z = np.array(porcent_pit)

Poly_pitch = PolynomialFeatures(degree=2)
model_pit = LinearRegression()
model_pit.fit(Poly_pitch.fit_transform(W),z)

# Entrenamiento del modelo de Regresión Lineal Múltiple para predecir la demora desde el retiro hasta la inducción para los batting

model_demora_bat = LinearRegression()
model_demora_bat.fit(Px, np.array(demora_bat))

model_demora_pit = LinearRegression()
model_demora_pit.fit(W, np.array(demora_pit))

parametros_bat = [experience_bat,g_bat, war_bat,h_bat,hr_bat,ab_bat,ba_bat,rbi_bat,ops_bat,obp_bat]
parametros_pit = [experience_pit,g_pit,war_pit,gf_pit,era_pit,l_pit,bb_pit,w_pit,w_l_pit,ip_pit]

coefientes_bat = list(map(lambda x: (ml.coeficiente([x],y,2)*100), parametros_bat))
coefientes_pit = list(map(lambda y: (ml.coeficiente([y],z,2)*100), parametros_pit))

# Cantidad de inducciones por año

años = [ind for ind in df_total["induction"]]
first_year = min(años)
last_year = max(años)
count_for_year = [años.count(unico) for unico in list(range(first_year, last_year + 1))]

# MEJORES PITCHER Y BATEADORES #

names_pitcher= df_pitching.index.to_list()
era_pitcher= df_pitching['era'].to_list()
war_pitcher=  df_pitching["war_p"].to_list()
g_pitcher=  df_pitching["g"].to_list()
l_pitcher=  df_pitching["l"].to_list()
w_pitcher=  df_pitching["w"].to_list()
ip_pitcher=  df_pitching['ip'].to_list()
bb_pitcher=  df_pitching['bb'].to_list()
w_l_pitcher=  df_pitching['w_l'].to_list()
gf_pitcher=  df_pitching['gf'].to_list()
experience_pitcher=  df_pitching['years_of_experience'].to_list()
link_p = df_pitching["link"].to_list()
inicio_p = df_pitching["first_game"].to_list()
final_p = df_pitching["last_game"].to_list()
seasons_p = ml.range_in_lists(inicio_p,final_p)

name_batting= df_batting.index.to_list()
war_batting= df_batting["war"].to_list()
g_batting = df_batting["g_bat"].to_list()
h_batting= df_batting["h"].to_list()
hr_batting= df_batting["hr"].to_list()
ba_batting= df_batting["ba"].to_list()
ab_batting= df_batting["ab"].to_list()
rbi_batting= df_batting["rbi"].to_list()
obp_batting= df_batting["obp"].to_list()
ops_batting= df_batting["ops"].to_list()
experience_batting= df_batting["years_of_experience"].to_list()
link_b = df_batting["link"].to_list()
inicio_b = df_batting["first_game"].to_list()
final_b = df_batting["last_game"].to_list()
seasons_b = ml.range_in_lists(inicio_b,final_b)

df_batting_score= pd.DataFrame({
    "Jugador": name_batting,
    "WAR_Batting": war_batting,
    "HR": hr_batting,
    "BA": ba_batting,
    "OPS": ops_batting,
    "OBP": obp_batting,
    "RBI": rbi_batting,
    "H": h_batting,
    'G': g_batting,
    "Seasons": experience_batting
})

# Crear métricas por temporada
df_batting_score["HR_temp"] = df_batting_score["HR"] / df_batting_score["Seasons"]
df_batting_score["WAR_temp"] = df_batting_score["WAR_Batting"] / df_batting_score["Seasons"]
df_batting_score["RBI_temp"] = df_batting_score["RBI"] / df_batting_score["Seasons"]
df_batting_score["H_temp"] = df_batting_score["H"] / df_batting_score["Seasons"]
df_batting_score["G_temp"] = df_batting_score["G"] / df_batting_score["Seasons"]

# Seleccionar métricas a normalizar
metricas_batting= ["HR_temp", "WAR_temp",'G_temp',"RBI_temp", "H_temp", "BA", "OPS", "OBP"]

# Normalizar con Min-Max 
scaler_batting= MinMaxScaler()
df_bat_norm = df_batting_score.copy()
df_bat_norm[metricas_batting] = scaler_batting.fit_transform(df_batting_score[metricas_batting])

# Calcular Score para Batting
df_bat_norm["Score"] = (
    0.4 * df_bat_norm["WAR_temp"] +
    0.1 * df_bat_norm["HR_temp"] +
    0.1 * df_bat_norm["RBI_temp"] +
    0.2 * df_bat_norm["G_temp"]+
    0.1 * df_bat_norm["OPS"] +
    0.1 * df_bat_norm["OBP"] +
    0.1 * df_bat_norm["BA"]
)

# Ranking para Batting
ranking_batting= df_bat_norm.sort_values("Score", ascending=False)

df_player_pitcher= pd.DataFrame({
    "Jugador": names_pitcher,
    "WAR_Pitching": war_pitcher,
    "ERA": era_pitcher,
    "BB": bb_pitcher,
    "IP": ip_pitcher,
    'G': g_pitcher,
    'W-L' : w_l_pitcher,
    'W': w_pitcher,
    'L': l_pitcher,
    'GF': gf_pitcher,
    "Seasons": experience_pitcher
})

df_player_pitcher["WAR_temp"] = df_player_pitcher["WAR_Pitching"] / df_player_pitcher["Seasons"]
df_player_pitcher["G_temp"] = df_player_pitcher["G"] / df_player_pitcher["Seasons"]
df_player_pitcher["BB_temp"] = df_player_pitcher["BB"] / df_player_pitcher["Seasons"]
df_player_pitcher["IP_temp"] = df_player_pitcher["IP"] / df_player_pitcher["Seasons"]
df_player_pitcher["W_temp"] = df_player_pitcher["W"] / df_player_pitcher["Seasons"]
df_player_pitcher["L_temp"] = df_player_pitcher["L"] / df_player_pitcher["Seasons"]
df_player_pitcher["GF_temp"] = df_player_pitcher["GF"] / df_player_pitcher["Seasons"]

metricas_pitcher= ["WAR_temp",'G_temp', "ERA","BB_temp","IP_temp",'W-L','W_temp','L_temp','GF_temp']

scaler_pitcher= MinMaxScaler()
df_pit_norm = df_player_pitcher.copy()
df_pit_norm[metricas_pitcher] = scaler_pitcher.fit_transform(df_pit_norm[metricas_pitcher])

df_pit_norm["Score"] = (
    0.3 * df_pit_norm["WAR_temp"] +
    0.2 * df_pit_norm['G_temp'] +
    0.1 * df_pit_norm["BB_temp"] +
    0.4 *df_pit_norm["IP_temp"]+
    0.2 *df_pit_norm['W-L'] +
    0.1 *df_pit_norm['GF_temp']
)

ranking_pitcher= df_pit_norm.sort_values("Score",ascending=False)

#Análisis de los Pitchers

df_pitchings = df_players.dropna(subset=["g"])
pitchers = df_total.dropna(subset=["g"])['inducted_as'].to_list()
porcents = df_total.dropna(subset=["g",'% of Ballots']).sort_values(by='% of Ballots',ascending=False).index

pitch_players = pitchers.count('Player')
pitch_manager = pitchers.count('Manager')
pitch_executive = pitchers.count('Pioneer/Executive')
pitch_umpire = pitchers.count('Umpire')

df_players_ambos = df_players.copy().dropna(subset=['g', 'g_bat','w', 'war_p', '% of Ballots', 'years_of_waiting_to_enter'])

df_players_ambos['Total_de_Juegos'] = df_players_ambos[['g_bat','g']].sum(axis=1,skipna=True)
df_players_ambos['Total_de_WAR'] = df_players_ambos[['war','war_p']].sum(axis=1,skipna=True)

game_ambos = df_players_ambos['Total_de_Juegos'].to_list()
war_ambos = df_players_ambos['Total_de_WAR']
experience_ambos = df_players_ambos['years_of_experience']
ab_ambos = df_players_ambos['ab']
ba_ambos = df_players_ambos['ba']
h_ambos = df_players_ambos['h']
hr_ambos = df_players_ambos['hr']
bb_ambos = df_players_ambos['bb']
rbi_ambos = df_players_ambos['rbi']
obp_ambos = df_players_ambos['obp']
ops_ambos = df_players_ambos['ops']
era_ambos = df_players_ambos['era']
gf_ambos = df_players_ambos['gf']
l_ambos = df_players_ambos['l']
w_ambos = df_players_ambos['w']
w_l_ambos = df_players_ambos['w_l']
ip_ambos = df_players_ambos['ip']
percent_ambos = df_players_ambos['% of Ballots']
demora_ambos = df_players_ambos['years_of_waiting_to_enter']

aspirantes_hof = ml.read_json("aspirantes_a_hof.json")

aspirantes = []

for i in range(1936, 2026):
   aspirantes.extend(ml.read_json("aspirantes_a_hof.json")[f'{i}'])

intentos_ambos = list(map(lambda x : aspirantes.count(x), df_players_ambos.index.to_list()))

# Función principal

def main() -> None:
    st.title(title)
    st.image(salón)
    st.write("Ya falta poco para el anuncio oficial por el presidente del Salón de la Fama del Baseball de Cooperstown de los " \
    "nuevos miembros del 2025 el próximo 27 de julio. Por lo que ahora los voy a sumergir en un análisis sobre los registros que alcanzaron" \
    " estos nuevos miembros y los inducidos en años anteriores que los llevaron a ser miembros del dicho salón en que se encuentran " \
    "jugadores, ejecutivos, managers y árbitros. Por lo que a continuación van a conocer los criterios para ser elegibles para en trar a" \
    " Cooperstown, los casos de éxitos que han hecho historia y detalles interesantes que sólo se llegan a conocer a travéz de los datos, " \
    "para que si usted es pelotero se embulle a alcanzar y superar los criterios de selección que se exige para ser exaltado en New York.")

    st.header("A continuación están los listados con los datos de los miembros actuales del Salón de la Fama del Baseball para que conozca los datos de estos exponentes del baseball para que los analice si desea.")
    df = st.selectbox("Selecciona la categoría que desee ver", list(categorias.keys()))
    st.dataframe(categorias[df])
    
    categoria = ['Player', 'Managers', 'Pioneer / Executive', 'Umpire']
    cantidad = [df_players.shape[0], df_managers.shape[0], df_Pioneer_Executive.shape[0], df_umpire.shape[0]]
    df_type = pd.DataFrame(
       {
          'Categoría': categoria,
          'Cantidad': cantidad
       }
    )

    player_p = df_players.dropna(subset=['g']).index.to_list()
    player_b = df_players.dropna(subset=['g_bat']).index.to_list()
    player_b.extend(player_p)

    manager_p = df_managers.dropna(subset=['g']).index.to_list()
    manager_b = df_managers.dropna(subset=['g_bat']).index.to_list()
    manager_b.extend(manager_p)

    Pioneer_Executive_p = df_Pioneer_Executive.dropna(subset=['g']).index.to_list()
    Pioneer_Executive_b = df_Pioneer_Executive.dropna(subset=['g_bat']).index.to_list()
    Pioneer_Executive_b.extend(Pioneer_Executive_p)

    umpire_p = df_umpire.dropna(subset=['g']).index.to_list()
    umpire_b = df_umpire.dropna(subset=['g_bat']).index.to_list()
    umpire_b.extend(umpire_p)

    players_nums = len(set(player_b))
    manager_nums = len(set(manager_b))
    Pioneer_Executive_nums = len(set(Pioneer_Executive_b))
    umpire_nums = len(set(umpire_b))

    col_type , col_type2 = st.columns(2)

    with col_type:
       count_type = px.pie(df_type, names='Categoría', values='Cantidad',color= "Categoría", color_discrete_sequence=[ '#33FF57','#FF5733', '#33C1FF', '#9D33FF'],  title='¿ Qué categoría tiene la mayor cantidad de miembros ?')
       count_type.update_layout(
       width=500,     
       height=500     
       )

       count_type.update_traces(
       textinfo='label+percent',
       textposition='inside',
       pull=[0, 0, 0.05, 0], 
       marker=dict(line=dict(color='white', width=2))
       )
       st.plotly_chart(count_type)

    with col_type2:
       df_with_nums = pd.DataFrame({"Categoría": ["Players", "Ejecutivos", "Managers", "Arbitros"], "Cantidades": [players_nums,Pioneer_Executive_nums,manager_nums,umpire_nums]})
       with_nums = px.bar(df_with_nums, x='Categoría', y='Cantidades', title='Cantidad de miembros por categoría con números')
       st.plotly_chart(with_nums)

    df_ind = pd.DataFrame({
    "Año": list(range(first_year, last_year + 1)),
    "Count": count_for_year
     })
    st.write('Empezando a hacer un poco de historia cada año son muchos los que han logrado entrar al salón de la fama del béisbol desde las pimeras ' \
    f'inducciones en 1936 hasta alcanzar la cifra de {len(df_total["link"].to_list())} inductos en el año actual, desde  las cantidades de inducciones oscilan cada año exceptuando algunas temporadas en que se han mantenido estables,' \
    ' en el perídodo de 1940 a 1960 hubieron'\
    ' cinco años en la cantidad de inductos fue nula, en 1988 se unió otro año a la lista y en 2021 se vuelve a repetir lo que no sucedía'\
    ' desde 1988 a causa de pandemia del Covid-19 que asechaba desde a finales de 2019 por lo no hubieron casi juegos de beisbol en 2020.'\
    'Como no todo es triste y , en 2006 hubo un pico en gráfica, fueron 12 fueros exaltados por el presidente del salón de la fama ese año, en los que se encuentran ' \
    'los cubanos José Méndez y Cristóbal Torriente, '\
    'la mayor cantidad desde sus inicios, por lo que aunque hubieron altas y bajas no se ha dejado marchitar la pasión por el béisbol. Y que les muestro la evidencia:')
    fig = px.line(df_ind, x="Año", y="Count", markers=True,
              title="Cantidad de inducciones por año",
              labels={"Count": "Cantidad de inducciones", "Año": "Año"},
              color_discrete_sequence=["green"])

    fig.update_traces(mode="lines+markers", hovertemplate="Año: %{x}<br>Valor: %{y:.2f}")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig)

    # Aspirantes a Entrar en El Salón de la Fama :

    st.subheader('¿ Son muchos los que han aspirado a entrar ?')

    aspirantes_hof = ml.read_json("aspirantes_a_hof.json")

    aspirantes = []

    for i in range(1936, 2026):
       aspirantes.extend(ml.read_json("aspirantes_a_hof.json")[f'{i}'])

    aspirantes_unicos = set(aspirantes)
    
    
    aspireantes_reiterados = [ i for i in aspirantes_unicos if aspirantes.count(i) > 1]
       
    aspirantes_for_year = [ len(aspirantes_hof[str(i)]) for i in range(1936, 2026)]

    df_aspirantes = pd.DataFrame(
       {
          "año": list(range(first_year, last_year + 1)),
          "cantidad": aspirantes_for_year
       }
    )

    fig_asp= px.line(df_aspirantes, x="año", y="cantidad", markers=True,
              title="Aspirantes a entrar al salón de la fama del béisbol por año",
              labels={"cantidad": "Cantidad", "Año": "Año"},
              color_discrete_sequence=["#9D33FF"])

    fig_asp.update_traces(mode="lines+markers", hovertemplate="Año: %{x}<br>Cantidad: %{y:.2f}")
    fig_asp.update_layout(hovermode="x unified")
    st.plotly_chart(fig_asp)

    inductos = len(df_total['link'].to_list())
    porciento = (inductos * 100) / len(aspirantes_unicos)
    
    st.write("Si observó detenidamente pudo observar que todos los aspirantes no han tenido la misma suerte ya que solamente el " \
    f"{round(porciento,2)} % de los que han provenido Asociación de Escritores de Béisbol de América (BBWAA), o por el Comité de Veteranos "\
    "han logrado entrar. Pero lo que tienen que estar ¡ Alertas ! los fanes del béisbol ya que desde los inicios del salón las cifras se " \
    "mantenían mayormente desde los 50  hasta superar la cifra de los 100 aspirantes en varios años, pero principalmente desde los años 60, " \
    "específicamente después del año 1967 las cifras no volvieron a alcanzar de la cifra de 80 candidatos, ni hablar después de 40 años en" \
    " que la cifras no han vuelto a rozar ni los 50, lo ya ha demostrado la falta interés por el deporte y la disminución de la calidad de " \
    "estos, con las nuevas tecnologías.Por lo que si la situación sigue así la cantidad de candidatos a postularse y la cantiddad de " \
    "peloteros que logren entrar puede ir de mal en peor en los próximos años. ")
    
    st.subheader("Si los que han logrado ser miembros son tan pocos, ¿ Cuáles son los criterios de selección para entrar a dicho salón ?")
    
    demora = df_total.copy().dropna(subset=['years_of_waiting_to_enter'])['years_of_waiting_to_enter'].to_list()

    st.write('Para elegir jugadores que son aptos existen varios criterios de elegebilidad, como haber estado retirado por más de cinco ' \
    'años, participación en más de 10 temporadas de las grandes ligas, tener una conducta ejemplar, tener buenos números, haber recibido ' \
    'al menos el 75 % de los votos de las papeletas por la Asociación de Escritores de Béisbol de América (BBWAA) y si no es elegido por esta vía puede ser considerado ' \
    'por el Cómite de Veteranos si tiene mucha experiencia, es decir bastantes años de retiros. En caso de los Mánagers, ejecutivos y ' \
    'arbitros anteriormente mensionados puede ser elegidos si tienen un impacto significativo en el béisbol en toda su carrera. Aunque no hay que tener miedo ya que ' \
    f' desde los inicios del salón de un total de {len(aspirantes_unicos)} aspirantes {len(aspireantes_reiterados)} han intentado entrar más de una vez y los '\
    f' que han podido entrar han tardado desde {min(demora)} años hasta {max(demora)} años en entrar pues los criterios de selección '\
    'son muy estrictos. ')

    st.subheader("Entonces hablando del pollo del arroz con pollo 'los números', ¿cuáles contribuyen en más para ser elegidos y cuáles " \
    "menos?")

    st.markdown("##### Comparación del aporte de cada parámetro de rendimiento para ayudan a alcanzar el 75 % de las boletas para los bateadores y los pitcher: ")
   
    bvsp = make_subplots(rows=1, cols=2,subplot_titles=["Influencia de cada parámetro en los bateadores",'Influencia de cada parámetro en los pitchers'])
    bvsp.add_trace(
    go.Bar(
    x= coefientes_bat,
    y=['Experiencia','Jugadas','WAR', 'Hits', 'HR', 'AB', 'BA', 'RBI', 'OPS', 'OBP' ],
    orientation='h'
    ),row=1, col=1)

    bvsp.add_trace(
    go.Bar(
    x= coefientes_pit,
    y=['Experiencia','Jugadas','WAR', 'Juegos terminados', 'ERA', 'Losses', 'Bases por bolas', 'Wines', 'W-%-L', 'Inning Pitching' ],
    orientation='h'
    )   ,row=1,col=2)
    bvsp.update_layout(showlegend=False)
    st.plotly_chart(bvsp)

    st.write("Como acabó de ver, ahora se dividió al grupo de  los jugadores entre bateadores y lanzadores, los números de los pitchers " \
    "tienden a contribuir mejor a un mayor por ciento en las votaciones, y esto se de debe aparentemente a la integridad de los pitchers ya que en el" \
    f" salón son miembros {len(pitchers)} pitchers, de ellos {pitch_players} jugadores, {pitch_executive} ejecutivos, {pitch_umpire} "\
    f"arbitros y {pitch_manager} managers. Por lo que al ser la cantidad de de pitchers en la categoría de jugadores altísima representando el "\
    f"{round((pitch_players*100)/len(df_total['link'].to_list()),2)} % de total "\
    "de miembros y hasta el 2025 el total de lanzadores también ejercieron en alguna etapa de su carrera la función de bateadores, lo que" \
    " resalta una calidad superior, por lo que si desea ser elegido podría parecer que es mejor que haga de todo en vez de establecerse en una sóla área del " \
    "béisbol.")

    df_war_vs_games = df_players.copy()
    
    df_war_vs_games['Total_de_Juegos'] = df_war_vs_games[['g_bat','g']].sum(axis=1,skipna=True)
    df_war_vs_games['Total_de_WAR'] = df_war_vs_games[['war','war_p']].sum(axis=1,skipna=True)

    df_demora = df_war_vs_games.copy().dropna(subset=['years_of_waiting_to_enter'])


    war_b = df_war_vs_games.dropna(subset=['war'])['war'].to_list()
    war_p = df_war_vs_games.dropna(subset=['war_p'])['war_p'].to_list()

    g_b = df_war_vs_games.dropna(subset=['g_bat'])['g_bat'].to_list()
    g_p = df_war_vs_games.dropna(subset=['g'])['g'].to_list()
     
    coeficientes_b = np.polyfit(g_b, war_b, deg=5)
    fx = np.poly1d(coeficientes_b)
    eje_x_b = np.linspace(min(g_b), max(g_b), 1000)
    eje_y_b = fx(eje_x_b)

    coeficientes_p = np.polyfit(g_p, war_p, deg=5)
    gx = np.poly1d(coeficientes_p)
    eje_x_p = np.linspace(min(g_p), max(g_p), 1000)
    eje_y_p = gx(eje_x_p)

    gvsw = go.Figure()

    gvsw.add_trace(go.Scatter(
    x=g_b, y=war_b,
    mode="markers",
    name="Batting",
    marker=dict(color="royalblue")
    ))

    gvsw.add_trace(go.Scatter(
    x=g_p, y=war_p,
    mode="markers",
    name="Pitcher",
    marker=dict(color="red")
    ))
    
    gvsw.add_trace(go.Scatter(
    x=eje_x_b, y=eje_y_b,
    mode="lines",
    name="Curva Ajuste para los battings",
    line=dict(color="firebrick", width=3)
    ))
    
    gvsw.add_trace(go.Scatter(
    x=eje_x_p, y=eje_y_p,
    mode="lines",
    name="Curva Ajuste para los pitchers",
    line=dict(color="blue", width=3)
    ))

    gvsw.update_layout(
    title="¿ Entonces será cierto que un lanzador contribuyen más a la victoria de su equipo que un bateador ?",
    xaxis_title="Games",
    yaxis_title="Wins Above Replacement",
    template="plotly_dark",      
    font=dict(family="Arial", size=14),
    hovermode="x unified"
    )
    st.plotly_chart(gvsw)

    st.image(Image.open('imagenes/flecha_roja.png'), width=300)
    st.subheader('¡NO!')
    st.write('Los bateadores tienen una mayor tendencia a mayor tendencia a aportar más victorias a su equipo según la cantidad de juegos jugados que los pitchers, ' \
    'aunque estos últimos tiendan a ocupar una mejor posición en el ranking debido a que también jan ejercido como ' \
    'bateadores a los largo de su carrera. De los cuales han salido las primeras estrellas de la historia del béisbol ' \
    'miembros del salón de la fama que permanecen los Top 10 de los mejores bateadores y  mejores lanzadores.')
    
    st.subheader("¿ Quiénes son los peloteros más éxitosos y peores inductos en el salón de la fama ?")
    image1, image2, image3 = st.columns(3)
    with image1:
     st.image(placas,caption='Placas de los mejores', width=200)
    with image2:
       st.image(Babe_Ruth, caption='Babe Ruth' ,width=200)
    with image3:
       st.image(Mariano_Rivera, caption='Mariano Rivera', width=200) 
       
    condición = st.selectbox("Selecciona las condiciones: ", ['Mejores', 'Peores'])
    
    if condición == 'Mejores':
     col_num, col_porcent = st.columns(2)

     with col_num:
       st.subheader('Según números')
       col_bat, col_pit = st.columns(2)
       with col_bat:
          st.subheader('Top 10 de los Batt')
          st.dataframe(pd.DataFrame({'Jugador': df_bat_norm.sort_values(by="Score", ascending=False)['Jugador'].to_list(), 'Score': df_bat_norm.sort_values(by="Score", ascending=False)['Score'].to_list()}).head(10).style.background_gradient(cmap="Blues"))
       with col_pit:
          st.subheader('Top 10 de los Pitch')
          st.dataframe(pd.DataFrame({'Jugador': df_pit_norm.sort_values(by="Score", ascending=False)['Jugador'].to_list(), 'Score': df_pit_norm.sort_values(by="Score", ascending=False)['Score'].to_list()}).head(10).style.background_gradient(cmap="Greens"))

     with col_porcent:
       st.subheader('Según Votos')
       col_bat, col_pit = st.columns(2)
       with col_bat:
          st.subheader('Top 10 de los Batt')
          df_percent_bat = df_batting.dropna(subset=['% of Ballots']).sort_values(by='% of Ballots',ascending=False)
          st.dataframe(pd.DataFrame({'Jugador': df_percent_bat.index.to_list(), 'Percent':  df_percent_bat['% of Ballots'].to_list()}).head(10).style.background_gradient(cmap="Oranges"))
       with col_pit:
          st.subheader('Top 10 de los Pitch')
          df_percent_pit = df_pitching.dropna(subset=['g','% of Ballots']).sort_values(by='% of Ballots',ascending=False)
          st.dataframe(pd.DataFrame({'Jugador': df_percent_pit.index.to_list(), 'Percent':  df_percent_pit['% of Ballots'].to_list()}).head(10).style.background_gradient(cmap="Purples"))
   
    if condición == 'Peores':
      col_num, col_porcent = st.columns(2)
      with col_num:
       st.subheader('Según números')
       col_bat, col_pit = st.columns(2)
       with col_bat:
          st.subheader('Top 10 de los Batt')
          st.dataframe(pd.DataFrame({'Jugador': df_bat_norm.sort_values(by="Score")['Jugador'].to_list(), 'Score': df_bat_norm.sort_values("Score")['Score'].to_list()}).head(10).style.background_gradient(cmap="Blues"))
       with col_pit:
          st.subheader('Top 10 de los Pitch')
          st.dataframe(pd.DataFrame({'Jugador': df_pit_norm.sort_values(by="Score")['Jugador'].to_list(), 'Score': df_pit_norm.sort_values("Score")['Score'].to_list()}).head(10).style.background_gradient(cmap="Greens"))

      with col_porcent:
       st.subheader('Según Votos')
       col_bat, col_pit = st.columns(2)
       with col_bat:
          st.subheader('Top 10 de los Batt')
          df_percent_bat = df_batting.dropna(subset=['g_bat','% of Ballots']).sort_values(by='% of Ballots')
          st.dataframe(pd.DataFrame({'Jugador': df_percent_bat.index.to_list(), 'Percent':  df_percent_bat['% of Ballots'].to_list()}).head(10).style.background_gradient(cmap="Oranges"))
       with col_pit:
          st.subheader('Top 10 de los Pitch')
          df_percent_pit = df_pitching.dropna(subset=['g','% of Ballots']).sort_values(by='% of Ballots')
          st.dataframe(pd.DataFrame({'Jugador': df_percent_pit.index.to_list(), 'Percent':  df_percent_pit['% of Ballots'].to_list()}).head(10).style.background_gradient(cmap="Purples"))
   
       
    st.subheader('¿ Cuántos jugadores hubieran podido entrar por Votación con al menos el  75 % y han sufrido de injusticia ?')
    
    df_p_sin_votos = df_pitching[df_pitching['% of Ballots'].isna()]
    
    df_b_sin_votos = df_batting[df_batting['% of Ballots'].isna()]

    era_p_votos = df_p_sin_votos['era'].to_list()
    war_p_votos =  df_p_sin_votos["Total_de_WAR"].to_list()
    g_p_votos =  df_p_sin_votos["Total_de_Juegos"].to_list()
    l_p_votos =  df_p_sin_votos["l"].to_list()
    w_p_votos =  df_p_sin_votos["w"].to_list()
    ip_p_votos =  df_p_sin_votos['ip'].to_list()
    bb_p_votos =  df_p_sin_votos['bb'].to_list()
    w_l_p_votos =  df_p_sin_votos['w_l'].to_list()
    gf_p_votos =  df_p_sin_votos['gf'].to_list()
    experience_p_votos =  df_p_sin_votos['years_of_experience'].to_list()

    war_b_votos = df_b_sin_votos["war"].to_list()
    g_b_votos = df_b_sin_votos["g_bat"].to_list()
    h_b_votos = df_b_sin_votos["h"].to_list()
    hr_b_votos = df_b_sin_votos["hr"].to_list()
    ba_b_votos = df_b_sin_votos["ba"].to_list()
    ab_b_votos = df_b_sin_votos["ab"].to_list()
    rbi_b_votos = df_b_sin_votos["rbi"].to_list()
    obp_b_votos = df_b_sin_votos["obp"].to_list()
    ops_b_votos = df_b_sin_votos["ops"].to_list()
    slg_b_votos = df_b_sin_votos["slg"].to_list()
    experience_b_votos = df_b_sin_votos["years_of_experience"].to_list()

    cantidad_b = len (experience_b_votos)
    cantidad_p = len(experience_p_votos)

    pitcher_actos = 0
    batting_actos = 0

    for p in range(cantidad_p):
       percent_p = model_pit.predict(Poly_pitch.fit_transform(np.array([[experience_p_votos[p], g_p_votos[p],gf_p_votos[p],war_p_votos[p],era_p_votos[p], l_p_votos[p], bb_p_votos[p], w_p_votos[p], w_l_p_votos[p], ip_p_votos[p]]])))
       if percent_p >= 75:
          pitcher_actos += 1

    for b in range(cantidad_b):
       percent_b = model_bat.predict(Poly_bat.fit_transform(np.array([[experience_b_votos[b], g_b_votos[b],war_b_votos[b],h_b_votos[b],hr_b_votos[b],ab_b_votos[b],ba_b_votos[b],rbi_b_votos[b],ops_b_votos[b],obp_b_votos[b], slg_b_votos[b]]])))
       if percent_b >= 75:
          batting_actos += 1

    df_inj = pd.DataFrame({'Categoría': ['Pitchers', 'Battings'], 'Cantidad': [pitcher_actos, batting_actos]})
    fig_inj = px.bar(df_inj, x='Categoría', y='Cantidad',color_discrete_sequence=["#E77B32", "#E77B32"], title='Cantidad de lanzadoresy bateadores que puediero haber alcanzado el 75 % de los votos')
    st.plotly_chart(fig_inj)

    st.subheader('¿ Cuál es el futuro del salón de la fama del béisbol de Cooperstown ? ')
    st.write(' El futuro del salón de la fama es incierto y impredecible ya que los datos muestran un decrecimiento de la cantida de aspirantes y en las tasas de acertación en el salón de la fama.' \
    ' Lo que se hace un llamado a no dejar morir el beísbol en que varios paises principalmente de latinoamerica ha ' \
    'llegado a ser deporte nacional. Si es pelotero o aficionado a dicho deporte entrene para llegar ha dicho salón de la fama, para lo cual debe jugar bastantes juegos para aportar más a la victoria a su equipo ' \
    ', además de que debe priorizar las bases por bolas, hits y bastantes juegos terminados sin ser reemplazado para tener más oprtunidades de destacar , ya que estos son los parámetros que hasta ahora ' \
    'más contribuyen a un mayor suerte en las votaciones. En cuanto a las principales figuras del baseball, el' \
    ' salón de la fama más que darle reconcimiento por sus números y hábilidades, hace que sus nombres brillen eternamente ' \
    'en los salones del salón de la fama siendo orgullo para futuros amantes del baseball, desde niños hasta ancianos arrepentidos por lo que un día pudieron hacer y no hicieron.')

if __name__ == "__main__":
    main()



