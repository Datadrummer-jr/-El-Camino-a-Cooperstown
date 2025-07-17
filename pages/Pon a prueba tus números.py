<<<<<<< HEAD
<<<<<<<< HEAD:pages/Pon a prueba tus números.py
import Camino_a_Cooperstown as app
import streamlit as st
import numpy as np
from PIL import Image

logo = Image.open("logo.jpg")

st.set_page_config(
    page_title="Pon Tus Números A Cooperstown",
    page_icon=logo,
    layout="wide")

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
          slg_b_form = st.number_input('SLG',step=0.001, format="%.3f")
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
          porcent_b = float(app.model_bat.predict(app.Poly_bat.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form,slg_b_form]]))))
          demora_b = float(app.model_demora_bat.predict(app.Poly_bat.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form, slg_b_form]]))))
          if porcent_b >= 75:
           st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_b)} años en entrar después de su retiro.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
       elif opción == "Pitching":
          porcent_p = float(app.model_pit.predict(app.Poly_pitch.fit_transform(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]]))))
          demora_p = float(app.model_demora_pit.predict(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]])))
          if porcent_p >= 75:
            st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_p)} años en entrar.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
   
if __name__ == "__main__":
========
=======
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
import Camino_a_Cooperstown as app
import streamlit as st
import numpy as np
from PIL import Image

logo = Image.open("logo.jpg")

st.set_page_config(
<<<<<<< HEAD
    page_title="Bienvenido a Cooperstown",
=======
    page_title="Pon Tus Números A Cooperstown",
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
    page_icon=logo,
    layout="wide")

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
<<<<<<< HEAD
=======
          slg_b_form = st.number_input('SLG',step=0.001, format="%.3f")
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
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
<<<<<<< HEAD
          porcent_b = float(app.model_bat.predict(app.Poly.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form]]))))
          demora_b = float(app.model_demora_bat.predict(app.Poly.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form]]))))
=======
          porcent_b = float(app.model_bat.predict(app.Poly_bat.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form,slg_b_form]]))))
          demora_b = float(app.model_demora_bat.predict(app.Poly_bat.fit_transform(np.array([[experience_b_form,g_b_form,war_b_form,h_b_form,hr_b_form,ab_b_form,ba_b_form,rbi_b_form,ops_b_form,obp_b_form, slg_b_form]]))))
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
          if porcent_b >= 75:
           st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_b)} años en entrar después de su retiro.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_b,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
       elif opción == "Pitching":
<<<<<<< HEAD
          porcent_p = float(app.model_pit.predict(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]])))
=======
          porcent_p = float(app.model_pit.predict(app.Poly_pitch.fit_transform(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]]))))
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
          demora_p = float(app.model_demora_pit.predict(np.array([[experience_p_form,g_p_form,gf_p_form,war_p_form,era_p_form,l_p_form,bb_p_form,w_p_form,W_L_p_form,ip_p_form]])))
          if porcent_p >= 75:
            st.success(f"¡Muchas Felicidades! Se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo que podría entrar en el salón de la fama de béisbol y tardaría aproximadamente {int(demora_p)} años en entrar.")
          else:
             st.error(f"Lo siento, se predice que según tus datos aportados las boletas serían de un {round(porcent_p,2)} %, por lo no podría entrar en el salón de la fama de béisbol.")
<<<<<<< HEAD

if __name__ == "__main__":
>>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733:pages/Pon-A-Prueba-Tus-Números.py
=======
   
if __name__ == "__main__":
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
    main()