
from typing import List
import numpy as np

def you_type(cadena: str):
  n = len(cadena)
  caracteres = [l for l in cadena]
  if caracteres[0] == "." and cadena[1:n].isdigit():
    return float(cadena)
  if cadena.isdigit():
    return int(cadena)
  if cadena[0] == "-":
    return float(cadena.strip())
  else:
    for i in cadena:
      if "." in caracteres and i.isdigit():
        return float(cadena)
      else:
        return str(cadena)
      
def index_minimo(lista: List[int], origen: List) -> int:
    origen.append(lista.index(min(lista)))

def index_maximo(lista: List[int], origen: List) -> int:
    origen.append(lista.index(max(lista)))

def max_valor(matrix: List[List[int]]) -> List[List[int]]:
  matrix = np.array(matrix)
  traspuesta = matrix.T
  sumas = []
  for i in range(len(traspuesta)):
    sumas.append(float(sum(list(filter(lambda x: x!= None , traspuesta[i])))))
  maximo = max(sumas)
  return sumas.index(maximo)

def min_valor(matrix: List[List[int]]) -> List[List[int]]:
  matrix = np.array(matrix)
  traspuesta = matrix.T
  sumas = []
  for i in range(len(traspuesta)):
    sumas.append(float(sum(list(filter(lambda x: x!= None , traspuesta[i])))))
  maximo = min(sumas)
  return sumas.index(maximo)


def my_protly(x: List[int], y: List[int], line: str, title: str, eje_x : str, eje_y: str, name_legend: str, colors : str= "royalblue") -> None:
  import plotly.graph_objects as go
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name= line, line=dict(color=colors)))
  fig.update_layout(
    title=title,
    xaxis_title=eje_x,
    yaxis_title=eje_y,
    legend_title=name_legend
  )
  return fig

def my_protly_double_y(x: List[int], y1: List[int], y2: List[int], line1: str, line2: str, title: str, eje_x : str, eje_y: str, name_legend: str) -> None:
  import plotly.graph_objects as go
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name= line1))
  fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name= line2))
  fig.update_layout(
    title=title,
    xaxis_title=eje_x,
    yaxis_title=eje_y,
    legend_title=name_legend
  )
  return fig



