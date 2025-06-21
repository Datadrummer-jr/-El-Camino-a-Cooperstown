import streamlit as st
from streamlit_navigation_bar import st_navbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from statistics import mode
import my_library as ml
from streamlit_carousel import carousel

logo = Image.open("logo.jpg")

st.set_page_config(
    page_title="Mi Aplicación",
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


d = {'x': [-4,-3,-2,-1, 0, 1, 2,3,4], 'y': [16,9,4,1,0,1, 4,9,16]}
dfp = pd.DataFrame(data=d)
pd.DataFrame()
# Crear gráfico interactivo con Plotly
fig = px.line(dfp, x="x", y="y", title="Gráfico Interactivo de p(x)")
#fog , ax = plt.subplots()

# Mostrar gráfico en Streamlit
#st.title("Visualización con Plotly en Streamlit")
#st.pyplot(fog)
#st.plotly_chart(fig)


bat = df_players.T
names = [i for i in bat[:16]]

maximos = []
minimos = []

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

batting = [war, h,hr,ba,ab,rbi,obp,ops,slg]

for minimo in batting:
    ml.index_minimo(minimo,minimos)

for maximo in batting:
    ml.index_maximo(maximo,maximos)


moda_minima = mode(minimos)
 
new_df = df_players.shape[1]


for i in df_players.T:
    names_batting.append(i)

#Henry_Aaron_image = Image.open("aaron.jpg")

máximo = link[ml.max_valor(batting)]
mínimo = link[ml.min_valor(batting)]


best_pitching= pd.read_json("best_batting.json")



def best_bat():
   year = []
   for i in range(first[ml.max_valor(batting)],last[ml.max_valor(batting)] +1 ):
      year.append(i)

def main():
    st.title("Bienvenido a Cooperstown")
    st.write("Ya falta poco para el anuncio oficial por el presidente del Salón de la Fama del Baseball de Cooperstown de los " \
    "nuevos miembros del 2025 el próxi:mo 27 de julio. Por lo que les comparto un análisis sobre los registros que alcanzaron estos nuevos miembros y los " \
    "inducidos en años anteriores que los llevaron a ser mibros del dicho salón en que se encuentran jugadores, ejecutivos, managers y árbitros.")
    st.header("A continuación están los listados con los datos de los miembros actuales del Salón de la Fama del Baseball")
    df = st.selectbox("", list(categorias.keys()))
    st.dataframe(categorias[df])
    st.subheader("¿ Se ha preguntado cual es bateador miembro del salón de la fama del baseball con los mejores números o que tiene los peores números?")
    img1 = Image.open("Hank_Aaron_1960.png")
    img2 = Image.open("Hank_Aaron_1974.jpg")
    img3 = Image.open("Hank_Aaron_on_October_26,_2016.jpg")
    img4 = Image.open("HankAaronHallofFamePlaque.jpg")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.image(img1, caption="Henry Aaron en 1960", width= 250)

    with col2:
      st.image(img2, caption="Henry Aaron en 1974", width=300)

    with col3:
      st.image(img3, caption="Henry Aaron en 2016", width=300)

    with col4:
      st.image(img4, caption="tarja de Henry Aaron", width= 300)
 
    best_bat()

if __name__ == "__main__":
    main()

