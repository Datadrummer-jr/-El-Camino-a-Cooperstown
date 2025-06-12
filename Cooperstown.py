import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_json("hof.json")

pd.set_option("display.max_rows", None)

df_total = df.T

df_players = df.T[~df.T["inducted_as"].isin(["Manager", "Pioneer/Executive", "Umpire"])]

df_managers =  df.T[~df.T["inducted_as"].isin(["Player", "Pioneer/Executive", "Umpire"])]

df_Pioneer_Executive =  df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Umpire"])]

df_umpire = df.T[~df.T["inducted_as"].isin(["Player", "Manager", "Pioneer/Executive"])]

logo = Image.open("logo.jpg")


st.set_page_config(
    page_title="Mi Aplicación",
    page_icon=logo,
    layout="wide"
)

categorias = {
    "Jugadores": df_players,
    "Managers": df_managers,
    "Ejecutivos": df_Pioneer_Executive,
    "Árbitros": df_umpire,
    "todos" : df_total
}

def main():
    st.title("Bienvenido a Cooperstown")
    st.write("Ya falta poco para el anuncio oficial por el presidente del Salón de la Fama del Baseball de Cooperstown de los " \
    "nuevos miembros del 2025 el próximo 27 de julio. Por lo que les comparto un análisis sobre los registros que alcanzaron estos nuevos miembros y los " \
    "inducidos en años anteriores que los llevaron a ser mibros del dicho salón en que se encuentran jugadores, ejecutivos, managers y árbitros.")
    st.header("A continuación están los listados con los datos de los miembros actuales del Salón de la Fama del Baseball")
    df = st.selectbox("Selecciona una categoría de miembro: ", list(categorias.keys()))
    st.dataframe(categorias[df]) 
if __name__ == "__main__":
    main()

