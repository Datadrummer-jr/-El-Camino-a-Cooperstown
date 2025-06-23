import streamlit as st
from streamlit_navigation_bar import st_navbar
import pandas as pd
import json 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import my_library as ml
from streamlit_carousel import carousel

logo = Image.open("logo.jpg")

st.set_page_config(
    page_title="Bienvenido a Cooperstown",
    page_icon=logo,
    layout="wide"
)

df = pd.read_json("hof.json")

pd.set_option("display.max_rows", None)

df_total = df.T

df_players = df.T[~df.T["inducted_as"].isin(["Manager", "Pioneer/Executive", "Umpire"])]

df_managers =  df.T[~df.T["inducted_as"].isin(["Player", "Pioneer/Executive", "Umpire"])]

df_Pioneer_Executive =  df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Umpire"])]

df_umpire = df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Pioneer/Executive"])]

categorias = {
    "Jugadores": df_players,
    "Managers": df_managers,
    "Ejecutivos": df_Pioneer_Executive,
    "Árbitros": df_umpire,
    "todos" : df_total
}

bat = df_players.T
names_bat = [i for i in bat[:17]]

names_batting = []

link = df_players["link"].tolist()
war = df_players["war"].tolist()
ab = df_players["ab"].tolist()
h = df_players["h"].tolist()
hr = df_players["hr"].tolist()
ba = df_players["ba"].tolist()
rbi = df_players["rbi"].tolist()
obp = df_players["obp"].tolist()
ops = df_players["ops"].tolist()
slg = df_players["slg"].tolist()
first = df_players["first_game"].tolist()
last = df_players["last_game"].tolist()


w_p = df_players["w"].tolist()
war_pp = df_players["war_p"].tolist()
l_p = df_players["l"].tolist()
era_p = df_players["era"].tolist()
g_p = df_players["g"].tolist()


batting = [war, h,hr,ba,ab,rbi,obp,ops,slg]

 
new_df = df_players.shape[1]


for nombre in df_players.T:
    names_batting.append(nombre)

pitching = [w_p, war_pp, l_p, era_p, g_p]

with open("best_batting.json") as f:
    best_batting = json.load(f)

with open("best_pitching.json") as g:
    best_pitching = json.load(g)

bat_year = [ y for y in range(first[ml.max_valor(batting)],last[ml.max_valor(batting)] +1 )]

b_war = best_batting['war']
b_h = best_batting['h']
b_hr = best_batting['hr']
b_ba = best_batting['ba']
b_ab = best_batting['ab']
b_g = best_batting['g']

graph_b_war = ml.my_protly(bat_year,b_war,"WAR: Wins above replacement", f"Comportamiento del WAR por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Wins above replacement", "Leyenda")
graph_b_g = ml.my_protly(bat_year,b_g,"G: Jugadas ganadas", f"Comportamiento de las jugadas ganadas por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Jugadas Ganadas", "Leyenda")
graph_b_h = ml.my_protly(bat_year,b_h,"H: Hits", f"Comportamiento de los Hits por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Hits", "Leyenda")
graph_b_hr = ml.my_protly(bat_year,b_hr,"HR: Home Runs", f"Comportamiento de los Home Runs por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Hmo Runs", "Leyenda")
graph_b_ab = ml.my_protly(bat_year,b_ab,"AB: At bats", f"Comportamiento de los AB por año de {names_batting[ml.max_valor(batting)]} según datos","Año","At bats", "Leyenda")
graph_b_ba = ml.my_protly(bat_year,b_ba,"BA: Hits / At bats", f"Comportamiento de los Hits / At bats por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Hits / At bats", "Leyenda")

para_batting = {
   "WAR": graph_b_war,
   "G": graph_b_g,
   "H": graph_b_h,
   "HR": graph_b_hr,
   "AB": graph_b_ab,
   "BA": graph_b_ba
}

pit_year = [z for z in range(first[ml.max_valor(pitching)],last[ml.max_valor(pitching)] +1 )]

p_war = best_pitching['war']
p_g = best_pitching['g']
p_w = best_pitching['w']
p_l = best_pitching['l']
p_era = best_pitching['era']

graph_p_war = ml.my_protly(pit_year,p_war, "WAR: Wins above replacement", f"Comportamiento del War por año de {names_batting[ml.max_valor(pitching)]} según datos","Año","Wins above replacement", "Leyenda", "red")
graph_p_g = ml.my_protly(pit_year,p_g,"G: Jugadas ganadas", f"Comportamiento de las jugadas ganadas por año de {names_batting[ml.max_valor(pitching)]} según datos","Año","Jugadas Ganadas", "Leyenda", "red")
graph_p_w = ml.my_protly(pit_year,p_w,"W: ", f"Comportamiento del W por año de {names_batting[ml.max_valor(pitching)]} según datos","Año","W", "Leyenda", "red")
graph_p_l = ml.my_protly(pit_year,p_l,"L: ", f"Comportamiento de las L por año de {names_batting[ml.max_valor(pitching)]} según datos","Año","L", "Leyenda", "red")
graph_p_era = ml.my_protly(pit_year,p_era,"ERA: earned_run_avg", f"Comportamiento de las earned_run_avg por año de {names_batting[ml.max_valor(pitching)]} según datos","Año","earned_run_avg", "Leyenda", "red")

para_pitching = {
   "WAR": graph_p_war,
   "G": graph_p_g,
   "W": graph_p_w,
   "L": graph_p_l,
   "ERA": graph_p_era
}

# Nombres de los Pitcher 

names_pit = [name for name in bat[18:].T.dropna().T]

# Variables Independientes para los pitcher

era_pit = []
war_pit = []
g_pit = []
l_pit = []
w_pit = []
experience_pit = []

#Variables independientes para los bateadores

war_bat = []
g_bat = []
h_bat = []
hr_bat = []
ba_bat = []
ab_bat = []
rbi_bat = []
obp_bat = []
ops_bat = []
slg_bat = []
experience_bat = []

#variable dependiente para los pitcher

porcent_pit = []

# Variable dependiente para los bateadores

porcent_bat = []

for numbers_pit in bat[18:].T.dropna().T:
   era_pit.append(bat[18:].T.dropna().T[numbers_pit].T["era"])
   g_pit.append(bat[18:].T.dropna().T[numbers_pit].T["g"])
   war_pit.append(bat[18:].T.dropna().T[numbers_pit].T["war_p"])
   l_pit.append(bat[18:].T.dropna().T[numbers_pit].T["l"])
   w_pit.append(bat[18:].T.dropna().T[numbers_pit].T["w"])

for exp in names_pit:
   experience_pit.append(bat.T.T[exp].T['years_of_experience'])
   porcent_pit.append(bat.T.T[exp].T['% of Ballots'])
   
for numbers in bat[:17].T.dropna().T:
   war_bat.append(bat[:17].T.dropna().T[numbers].T["war"])
   g_bat.append(bat[:17].T.dropna().T[numbers].T["g_bat"])
   h_bat.append(bat[:17].T.dropna().T[numbers].T["h"])
   hr_bat.append(bat[:17].T.dropna().T[numbers].T["hr"])
   ab_bat.append(bat[:17].T.dropna().T[numbers].T["ab"])
   ba_bat.append(bat[:17].T.dropna().T[numbers].T["ba"])
   rbi_bat.append(bat[:17].T.dropna().T[numbers].T["rbi"])
   ops_bat.append(bat[:17].T.dropna().T[numbers].T["ops"])
   obp_bat.append(bat[:17].T.dropna().T[numbers].T["obp"])
   experience_bat.append(bat[:17].T.dropna().T[numbers].T["years_of_experience"])
   porcent_bat.append(bat[:17].T.dropna().T[numbers].T["% of Ballots"])

# Entrenamiento del modelo de Regresión Lineal Múltiple para predecir el por ciento de las boletas para los batting

X = np.array([experience_bat,g_bat, war_bat,h_bat,hr_bat,ab_bat,ba_bat,rbi_bat,ops_bat,obp_bat]).T

y = np.array(porcent_bat)

Poly = PolynomialFeatures(degree=2)
Px = Poly.fit_transform(X)

model_bat = LinearRegression()
model_bat.fit(Px,y)
#st.write("coeficiente:" , model_bat.score(Px, y))

# Función principal

def main():
    st.title("Bienvenido a Cooperstown")
    st.write("Ya falta poco para el anuncio oficial por el presidente del Salón de la Fama del Baseball de Cooperstown de los " \
    "nuevos miembros del 2025 el próximo 27 de julio. Por lo que ahora s voy a sumergir en un análisis sobre los registros que alcanzaron estos nuevos miembros y los " \
    "inducidos en años anteriores que los llevaron a ser mibros del dicho salón en que se encuentran jugadores, ejecutivos, managers y árbitros.")
    st.header("A continuación están los listados con los datos de los miembros actuales del Salón de la Fama del Baseball")
    df = st.selectbox("Selecciona la categoría que desee ver", list(categorias.keys()))
    st.dataframe(categorias[df])
    st.subheader("¿ Se ha preguntado cual es bateador miembro del salón de la fama del baseball con los mejores números?")
    img1 = Image.open("Hank_Aaron_1960.png")
    img2 = Image.open("Hank_Aaron_1974.jpg")
    img3 = Image.open("HankAaronHallofFamePlaque.jpg")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.image(img1, caption="Henry Aaron en 1960", width= 250)

    with col2:
      st.image(img2, caption="Henry Aaron en 1974", width=300)

    with col3:
      st.image(img3, caption="tarja de Henry Aaron", width= 300)
    
    bb = st.selectbox("Seleccione uno de algunos de los números del mejor bateador a lo largo de su carrera que desee ver:" , list(para_batting.keys()))
    st.plotly_chart(para_batting[bb])

    st.subheader("¿Y entonces que sucederá con los pitcher?" )

    img4 = Image.open("Cy_Young.jpg")
    img5 = Image.open("Cy_Young_by_Conlon,_1911-crop.jpg")
    img6 = Image.open("Cy_Young_HOF_plaque.jpg")

    col4, col5, col6 = st.columns(3)

    with col4:
       st.image(img4, "Cy Young joven",width=300)
    with col5:
       st.image(img5, "Cy Young en 1911", width=300)
    with col6:
       st.image(img6, "Placa de Cy Young", width=300)

    bp = st.selectbox("Seleccione uno de algunos se los números del mejor pitcher miembro del salón de la fama del baseball:" , list(para_pitching.keys()))
    st.plotly_chart(para_pitching[bp])

    st.subheader('Llene el siguiente formulario con el perfil de un para que vea si tiene posibilidades de entrar en el salón de la fama de Cooperstown a partir de los datos ingresados:')
    with st.form("Perfil de jugador"):
       posicion = st.selectbox('Seleccione la posicón:' , ['pitcher', 'batting'])
       g_b_form = st.number_input("Jugadas", step=1) 
       war_b_form = st.number_input("Wins above replacement",step=0.001, format="%.3f")
       h_b_form = st.number_input("Hits", step=1)
       hr_b_form = st.number_input("Home Runs", step=1)
       ab_b_form = st.number_input("At Bats", step=1)
       ba_b_form = st.number_input("Hits / At bats",step=0.001, format="%.3f")
       rbi_b_form = st.number_input('Runs Batted In',step=1)
       ops_b_form = st.number_input('onbase plus slugging',step=0.001, format="%.3f")
       obp_b_form = st.number_input('Onbase perce',step=0.001, format="%.3f")
       experience_b_form = st.number_input('Experiencia', step=1)
       send = st.form_submit_button('Predicir por ciento de las boletas')
    if send:
       if posicion == "batting":
          st.write(f"Se predice que según tus datos aportados las bolates serían de un {round((float(model_bat.predict(Poly.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form]]))))),2)} %")
       if posicion == "pitching":
          passs
if __name__ == "__main__":
    main()



