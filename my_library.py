
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go

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


def my_protly(x: List[int], y: List[int], line: str, title: str, eje_x : str, eje_y: str, name_legend: str, colors : str = "royalblue") -> None:
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name= line, line=dict(color=colors)))
  fig.update_layout(
    title=title,
    xaxis_title=eje_x,
    yaxis_title=eje_y,
    legend_title=name_legend
  )
  return fig

def balancear_matrix(matrix: List[List[any]],valor_para_balancear = 0):
  len_maximo = max([ len(v) for v in matrix])
  for g in matrix:
   if len(g) < len_maximo:
      diferencia = len_maximo - len(g)
      for _ in range(diferencia):
         g.append(valor_para_balancear)

def range_in_lists(lista1: List[int], lista2: List[int]) -> List[List[int]]:
  total_lists = []
  n = len(lista1)
  m = len(lista2)
  if n != m:
    return False
  def rango(indice: int):
    numbers = []
    for y in range(lista1[indice], lista2[indice] + 1):
      numbers.append(y)
    return numbers
  for i in range(n):
    total_lists.append(rango(i))
  return total_lists

def element_in_matrix(elemento: any, matrix: List[List[any]]) -> Tuple[int,int]:
  n = len(matrix)
  m = len(matrix[0])
  for i in range(n):
    for j in range(m):
      if matrix[i][j] == elemento:
        return (i,j)
      else:
        continue
      
