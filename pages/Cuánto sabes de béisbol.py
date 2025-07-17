<<<<<<< HEAD
import Camino_a_Cooperstown as app
import streamlit as st
import numpy as np
from PIL import Image

logo = Image.open("logo.jpg")

title="¿Cuánto sabes de béisbol?"

st.set_page_config(
    page_title=title,
    page_icon=logo,
    layout="wide")

def main():
    st.title(title)

if __name__ == "__main__":
=======
import Camino_a_Cooperstown as app
import streamlit as st
import numpy as np
from PIL import Image

logo = Image.open("logo.jpg")

title="¿Cuánto sabes de béisbol?"

st.set_page_config(
    page_title=title,
    page_icon=logo,
    layout="wide")

def main():
    st.title(title)

if __name__ == "__main__":
>>>>>>> 5b69f133b66fd4280b235a97a2054f5d118aa733
    main()