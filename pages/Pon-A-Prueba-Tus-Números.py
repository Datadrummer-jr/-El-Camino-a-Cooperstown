
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_json("hof.json")

pd.set_option("display.max_rows", None)

df_total = df.T

df_players = df.T[~df.T["inducted_as"].isin(["Manager", "Pioneer/Executive", "Umpire"])]

columnas_pit = ["era", "war_p","g" ,"l","w",'ip','bb','w_l','years_of_experience','gf',"% of Ballots", "war"]
df_pitch = df_players.dropna(subset=columnas_pit)

df_pitch['Total_de_Juegos'] = df_pitch[['g','g_bat']].sum(axis=1,skipna=True)
df_pitch['Total_de_WAR'] = df_pitch[['war','war_p']].sum(axis=1,skipna=True)

columnas_bat = ['years_of_experience', "% of Ballots", "war", "g_bat", "h", "hr", "ba","ab" ,"rbi","obp","ops" ]
df_batt = df_players.dropna(subset=columnas_bat)

df_batt['Total_de_Juegos'] = df_batt[['g','g_bat']].sum(axis=1,skipna=True)
df_batt['Total_de_WAR'] = df_batt[['war','war_p']].sum(axis=1,skipna=True)

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

X = np.array([experience_bat,g_bat, war_bat,h_bat,hr_bat,ab_bat,ba_bat,rbi_bat,ops_bat,obp_bat]).T

y = np.array(porcent_bat)

Poly = PolynomialFeatures(degree=2)

Px = Poly.fit_transform(X)

model_bat = LinearRegression()
model_bat.fit(Px,y)

W = np.array([experience_pit,g_pit, gf_pit,war_pit,era_pit,l_pit,bb_pit,w_pit,w_l_pit,ip_pit]).T

z = np.array(porcent_pit)

model_pit = LinearRegression()
model_pit.fit(W,z)

# Entrenamiento del modelo de Regresión Lineal Múltiple para predecir la demora desde el retiro hasta la inducción para los batting

model_demora_bat = LinearRegression()
model_demora_bat.fit(Px, np.array(demora_bat))

model_demora_pit = LinearRegression()
model_demora_pit.fit(W, np.array(demora_pit))

def main():
    st.title('Pon tus números a prueba, ¿ Te embullas ?')
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
          demora_b = float(model_demora_bat.predict(Poly.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form]]))))
          if porcent_b >= 75:
           st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_b)} años en entrar después de su retiro.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
       elif opción == "Pitching":
          porcent_p = float(model_pit.predict(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]])))
          demora_p = float(model_demora_pit.predict(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]])))
          if porcent_p >= 75:
            st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_p)} años en entrar.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")

if __name__ == "__main__":
    main()