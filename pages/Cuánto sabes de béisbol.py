import streamlit as st
from my_library import my_library as ml
import random
import pandas as pd
import time
from PIL import Image

st.markdown("""
    <style>
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease, transform 0.1s ease;
    }

    .stButton>button:hover {
        background-color: #1565C0;
        transform: scale(1.02);
    }

    .stButton>button:active {
        background-color: #0D47A1;
        transform: scale(0.98);
    }
    </style>
""", unsafe_allow_html=True)


logo = Image.open("imagenes/logo.jpg")

title="¿ Cuánto sabes de béisbol ?"

st.set_page_config(
    page_title=title,
    page_icon=logo,
    layout="wide")

# Puntuacione del juego
MAX_QUESTIONS = 10
WINNING_SCORE = 6


df_hof = ml.load_hof_data('hof.json')
df_countries = ml.load_country_data('nationality.json')

# Funciones auxiliares para el juego 

def get_random_player_data(df):
    """Selecciona un jugador aleatorio (una fila) del DataFrame."""
    if df.empty:
        return None, None
    player_row = df.sample(1).iloc[0]
    player_name = player_row["Name"]
    return player_name, player_row

def get_player_nationality(player_name, df_nat):
    """Obtiene la nacionalidad de un jugador desde el DataFrame de países."""
    if df_nat.empty:
        return "Desconocida"
    result = df_nat[df_nat["name"] == player_name]
    if not result.empty:
        return result.iloc[0]["country"]
    return "Desconocida"

def get_quiz_question(player_name, player_info):
    """Genera una pregunta aleatoria sobre un jugador y sus opciones."""
    # Lista de preguntas base
    question_types = [
        f"¿En qué año fue introducido al Salón de la Fama {player_name}?",
        f"¿Cuál fue el porcentaje de votos que recibió {player_name} para su inducción?",
        f"¿Cuántos jonrones (HR) conectó {player_name} en su carrera?",
        f"¿Cuál fue el promedio de bateo (BA) de {player_name}?",
        f"¿De qué nacionalidad es {player_name}?",
        f"¿Cuántas carreras impulsadas (RBI) tuvo {player_name}?",
        f"¿Cuál fue el WAR (Victorias Sobre Reemplazo) de {player_name} en su carrera?",
        f"¿Cuántos años de experiencia tuvo {player_name} en las Grandes Ligas?"
    ]
    
    # Añadir preguntas de lanzador solo si el jugador tiene estadísticas de pitcheo válidas
    if pd.notna(player_info.get('w')) and pd.notna(player_info.get('era')):
        question_types.append(f"¿Cuántos juegos ganó (W) {player_name} como lanzador?")
        question_types.append(f"¿Cuál fue el promedio de carreras limpias (ERA) de {player_name} ?")

    question_template = random.choice(question_types)
    question = question_template.format(player_name=player_name)

    correct_answer = "N/A"
    options = []

    if "porcentaje de votos" in question_template:
        correct_answer = player_info["% of Ballots"]
        options = generate_numeric_options(correct_answer, 3, 50.0, 100.0, decimal=True, precision=1)
    elif "año fue introducido" in question_template:
        correct_answer = int(player_info["induction"])
        options = generate_numeric_options(correct_answer, 3, 1900, 2025)
    elif "jonrones" in question_template:
        correct_answer = int(player_info["hr"])
        options = generate_numeric_options(correct_answer, 3, 100, 800)
    elif "promedio de bateo" in question_template:
        correct_answer = player_info["ba"]
        options = generate_numeric_options(correct_answer, 3, 0.200, 0.400, decimal=True, precision=3)
    elif "carreras impulsadas" in question_template:
        correct_answer = int(player_info["rbi"])
        options = generate_numeric_options(correct_answer, 3, 500, 2300)
    elif "WAR" in question_template:
        correct_answer = round(player_info["war"], 1)
        options = generate_numeric_options(correct_answer, 3, 20.0, 170.0, decimal=True, precision=1)
    elif "juegos ganó" in question_template:
        correct_answer = int(player_info["w"])
        options = generate_numeric_options(correct_answer, 3, 100, 520)
    elif "carreras limpias (ERA)" in question_template:
        correct_answer = player_info["era"]
        options = generate_numeric_options(correct_answer, 3, 1.50, 5.00, decimal=True, precision=2)
    elif "años de experiencia" in question_template:
        correct_answer = int(player_info["years_of_experience"])
        options = generate_numeric_options(correct_answer, 3, 10, 30)
    elif "nacionalidad" in question_template:
        correct_answer = get_player_nationality(player_name, df_countries)
        if correct_answer != "Desconocida":
             options = generate_nationality_options(correct_answer, df_countries, 3)
        else:
            return get_quiz_question(player_name, player_info)
    
    str_correct_answer = str(correct_answer)
    if str_correct_answer not in options:
        if options:
            options[random.randint(0, len(options) - 1)] = str_correct_answer
        else:
            options = [str_correct_answer]

    random.shuffle(options)
    return question, str_correct_answer, options

def generate_numeric_options(correct, num_options, min_val, max_val, decimal=False, precision=0):
    """Genera opciones numéricas aleatorias."""
    options = {correct}
    while len(options) < num_options + 1:
        if decimal:
            option = round(random.uniform(min_val, max_val), precision)
        else:
            option = random.randint(int(min_val), int(max_val))
        options.add(option)
    return [str(o) for o in options]

def generate_nationality_options(correct_nat, df_nat, num_options):
    """Genera opciones de nacionalidad aleatorias."""
    options = {correct_nat}
    all_nationalities = df_nat["country"].unique().tolist()
    available_options = [nat for nat in all_nationalities if nat != correct_nat]
    if len(available_options) >= num_options:
        options.update(random.sample(available_options, num_options))
    return list(options)

if 'score' not in st.session_state:
    st.session_state.score = 0
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_options' not in st.session_state:
    st.session_state.current_options = []
if 'correct_answer' not in st.session_state:
    st.session_state.correct_answer = None
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

def start_new_game():
    """Reinicia el estado del juego para empezar una nueva partida."""
    st.session_state.score = 0
    st.session_state.question_count = 0
    st.session_state.game_started = True
    st.session_state.game_over = False
    generate_new_question()

def generate_new_question():
    """Genera una nueva pregunta y actualiza el estado de la sesión."""
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.session_state.game_over = True
        return
    if df_hof.empty:
        st.warning("No hay datos de jugadores cargados para generar preguntas.")
        return
    player_name, player_info = get_random_player_data(df_hof)
    question, correct_answer, options = get_quiz_question(player_name, player_info)
    st.session_state.current_question = question
    st.session_state.current_options = options
    st.session_state.correct_answer = correct_answer

def check_answer(selected_option):
    """Verifica la respuesta, actualiza la puntuación y avanza a la siguiente pregunta."""
    st.session_state.question_count += 1
    
    if selected_option == st.session_state.correct_answer:
        st.session_state.score += 1
        st.success("¡Correcto! ✅")
    else:
        st.error(f"Incorrecto. La respuesta correcta era: **{st.session_state.correct_answer}** ❌")
    
    st.markdown("---")
    time.sleep(2)
    
    if st.session_state.question_count < MAX_QUESTIONS:
        generate_new_question()
    else:
        st.session_state.game_over = True
    
    st.rerun()

def main():
 st.title(title)

 if not st.session_state.game_started:
    st.write(f"¡Pon a prueba tus conocimientos sobre el Salón de la Fama! Responde {WINNING_SCORE} de {MAX_QUESTIONS} preguntas para ganar.")
    st.markdown("---")
    if st.button("🚀 Empezar Juego", use_container_width=True):
        start_new_game()
        st.rerun()
 elif st.session_state.game_over:
    st.subheader("¡Juego Terminado!")
    st.write(f"### Tu puntuación final es: **{st.session_state.score} / {MAX_QUESTIONS}**")
    
    if st.session_state.score >= WINNING_SCORE:
        st.balloons()
        st.success(f"🎉 ¡Felicidades! Has ganado. Conseguiste {st.session_state.score} respuestas correctas.")
    else:
        st.error(f"😞 No alcanzaste la puntuación mínima. Necesitabas como mínimo {WINNING_SCORE} preguntas correctas.")
    
    st.button("Jugar de Nuevo", on_click=start_new_game, use_container_width=True)
 else:
    st.markdown(f"**Pregunta {st.session_state.question_count + 1} de {MAX_QUESTIONS}**")
    st.progress((st.session_state.question_count) / MAX_QUESTIONS)
    st.markdown(f"**Puntuación actual: {st.session_state.score}**")
    st.markdown("---")
    
    if st.session_state.current_question:
        st.subheader(st.session_state.current_question)
        
        cols = st.columns(2)
        for i, option in enumerate(st.session_state.current_options):
            with cols[i % 2]:
                if st.button(option, key=f"opt_{st.session_state.question_count}_{i}", use_container_width=True):
                    check_answer(option)
    else:
        st.warning("Cargando pregunta...")
        st.button("Generar Nueva Pregunta", on_click=generate_new_question)

 st.markdown("---")
 st.write("¡Gracias por jugar!")

if __name__ == "__main__":
    main()
