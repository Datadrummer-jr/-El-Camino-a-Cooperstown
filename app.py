import streamlit as st
import pandas as pd
import json 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
import plotly.express as px
import my_library as ml

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
graph_b_hr = ml.my_protly(bat_year,b_hr,"HR: Home Runs", f"Comportamiento de los Home Runs por año de {names_batting[ml.max_valor(batting)]} según datos","Año","Home Runs", "Leyenda")
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
names_pit = [bat[18:].T.dropna().T]

columnas_pit = ["era", "war_p","g" ,"l","w",'ip','bb','w_l','years_of_experience','gf',"% of Ballots"]
df_pitch = df_players.dropna(subset=columnas_pit)

columnas_bat = ['years_of_experience', "% of Ballots", "war", "g_bat", "h", "hr", "ba","ab" ,"rbi","obp","ops" ]
df_batt = df_players.dropna(subset=columnas_bat)

# Variables Independientes para los pitcher

era_pit = df_pitch['era'].to_list()
war_pit =  df_pitch["war_p"].to_list()
g_pit =  df_pitch["g"].to_list()
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
experience_bat = df_batt["years_of_experience"].to_list()

#variable dependiente para los pitcher

porcent_pit =  df_pitch["% of Ballots"].to_list()

# Variable dependiente para los bateadores

porcent_bat = df_batt["% of Ballots"].to_list()


# Entrenamiento del modelo de Regresión Lineal Múltiple para predecir el por ciento de las boletas para los batting

X = np.array([experience_bat,g_bat, war_bat,h_bat,hr_bat,ab_bat,ba_bat,rbi_bat,ops_bat,obp_bat]).T

y = np.array(porcent_bat)

Poly = PolynomialFeatures(degree=2)
Px = Poly.fit_transform(X)

model_bat = LinearRegression()
model_bat.fit(Px,y)
#st.write("coeficiente:" , model_bat.score(Px, y))


W = np.array([experience_pit,g_pit, gf_pit,war_pit,era_pit,l_pit,bb_pit,w_pit,w_l_pit,ip_pit]).T

z = np.array(porcent_pit)

Gx = Poly.fit_transform(W)

model_pit = LinearRegression()
model_pit.fit(Gx,z)
#st.write("coeficiente:" , model_pit.score(Gx, z))

años = [ind for ind in df_total["induction"]]
first_year = min(años)
last_year = max(años)
count_for_year = [años.count(unico) for unico in list(range(first_year, last_year + 1))]

# Función principal

def main() -> None:
    st.title("Bienvenido a Cooperstown")
    st.write("Ya falta poco para el anuncio oficial por el presidente del Salón de la Fama del Baseball de Cooperstown de los " \
    "nuevos miembros del 2025 el próximo 27 de julio. Por lo que ahora s voy a sumergir en un análisis sobre los registros que alcanzaron estos nuevos miembros y los " \
    "inducidos en años anteriores que los llevaron a ser mibros del dicho salón en que se encuentran jugadores, ejecutivos, managers y árbitros.")

    st.header("A continuación están los listados con los datos de los miembros actuales del Salón de la Fama del Baseball para que conozca los datos de estos exponentes del baseball para que los analice si desea.")
    df = st.selectbox("Selecciona la categoría que desee ver", list(categorias.keys()))
    st.dataframe(categorias[df])

    st.subheader("Ahora veamos una gráfica con la comparación de la cantidad por cada categoría de miembro de este salón donde se apreciar que los jugadores se llevan todo en la escena como los protagonistas de cada juego:")
    categoria = ['Player', 'Managers', 'Pioneer / Executive', 'Umpire']
    cantidad = [df_players.shape[0], df_managers.shape[0], df_Pioneer_Executive.shape[0], df_umpire.shape[0]]
    df_type = pd.DataFrame(
       {
          'Categoría': categoria,
          'Cantidad': cantidad
       }
    )
    count_type = px.bar(df_type, x='Categoría', y='Cantidad',color= "Categoría", color_discrete_sequence=[ '#33FF57','#FF5733', '#33C1FF', '#9D33FF'],  title='Cantidad por tipo de miembro')
    st.plotly_chart(count_type)

    df_ind = pd.DataFrame({
    "Año": list(range(first_year, last_year + 1)),
    "Count": count_for_year
     })
    st.write('Empezando a hacer un poco de historia cada año son muchos los que entrar al salón de la fama desde las pimeras ' \
    'inducciones en 1936 donde las cantidades de inducciones oscilaban cada año,' \
    ' aunque en el perídodo de 1940 a 1960 hubieron'\
    ' cinco años en la cantidad de inductos fue nula, en 1988 se otro año a la lista y en 2021 se vuelve a repetir lo que no sucedía'\
    ' desde 1988 a causa de pandemia del Covid-19 que asechaba desde a finales de 2019 por lo no hubieron juego de beisbol en 2020.'\
    'Como no todo es triste, en 2006 hubo un pico en gráfica, fueron 12 fueros exaltados por el presidente del salón de la fama ese año, en los que se encuentran ' \
    'los cubanos José Méndez y Cristóbal Torriente, '\
    'la mayor cantidad desde sus inicios. Y que les muestro la evidencia:')
    fig = px.line(df_ind, x="Año", y="Count", markers=True,
              title="Cantidad de inducciones por año",
              labels={"Count": "Cantidad de inducciones", "Año": "Año"},
              color_discrete_sequence=["green"])

    fig.update_traces(mode="lines+markers", hovertemplate="Año: %{x}<br>Valor: %{y:.2f}")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig)

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

    st.subheader('Llene el siguiente formulario con el perfil de jugador para que vea si tiene posibilidades de entrar en el salón de la fama de Cooperstown a partir de los datos ingresados:')
    st.write('Nota: Para entrar al salón de la fama del baseball se requiere al menos el 75 % de votos de las papeletas.')
    opción = st.selectbox("¿Qué posición eliges?", ["Seleccione una posición...", "Batting", "Pitching"])
    
    with st.form("Perfil de Batting"):
        if opción == "Batting":
          g_b_form = st.number_input("Jugadas", step=1) 
          war_b_form = st.number_input("Wins above replacement",step=0.001, format="%.3f")
          h_b_form = st.number_input("Hits", step=1)
          hr_b_form = st.number_input("Home Runs", step=1)
          ab_b_form = st.number_input("At Bats", step=1)
          ba_b_form = st.number_input("Hits / At bats",step=0.001, format="%.3f")
          rbi_b_form = st.number_input('Runs Batted In',step=1)
          ops_b_form = st.number_input('onbase plus slugging',step=0.001, format="%.3f")
          obp_b_form = st.number_input('Onbase perce',step=0.001, format="%.3f")
          experience_b_form = st.number_input('Años de experiencia', step=1)
        elif opción == "Pitching":
         g_p_form = st.number_input("Jugadas", step=1) 
         war_p_form = st.number_input("Wins above replacement",step=0.001, format="%.3f")
         era_p_form = st.number_input("Earned run avg",step=0.001, format="%.3f" ) 
         bb_p_form = st.number_input("Bases por bola",step=1)
         gf_p_form = st.number_input("Jugadas terminadas",step=1)
         w_p_form = st.number_input("Wins",step=1)
         l_p_form = st.number_input("Losses",step=1)
         ip_p_form = st.number_input("Inning Pitching",step=0.001, format="%.3f" )
         W_L_p_form = st.number_input("Wins - losses Porcetange",step=0.001, format="%.3f" )
         experience_p_form = st.number_input('Años de experiencia', step=1)
        send = st.form_submit_button('Predicir por ciento de los votos de las papeletas para ser exaltado')
    if send:
       if opción == "Batting":
          porcent_b = float(model_bat.predict(Poly.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form]]))))
          if porcent_b >= 75:
           st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo que podría entrar en el salón de la fama de béisbol.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
       elif opción == "Pitching":
          porcent_p = float(model_pit.predict(Poly.fit_transform(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]]))))
          if porcent_p >= 75:
            st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo que podría entrar en el salón de la fama de béisbol.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")     

if __name__ == "__main__":
   main()






