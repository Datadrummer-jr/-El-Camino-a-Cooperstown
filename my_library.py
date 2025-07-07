
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def you_type(cadena: str):
  n = len(cadena)
  caracteres = [l for l in cadena]
  if cadena == "":
    return None
  if cadena.isdigit():
    return int(cadena)
  if caracteres[0] == "." and cadena[1:n].isdigit():
    return float(cadena)
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

def doble_y_protly(x: List[int], y: List[List[int]], line: List[str], colors : List[str], title: str, eje_x : str, eje_y: str, name_legend: str) -> None:
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y[0], mode='lines+markers', name= line[0], line=dict(color=colors[0])))
  fig.add_trace(go.Scatter(x=x, y=y[1], mode='lines+markers', name= line[1], line=dict(color=colors[1])))
  fig.update_layout(
    title=title,
    xaxis_title=eje_x,
    yaxis_title=eje_y,
    legend_title=name_legend
    )
  return fig

def balancear_matrix(matrix: List[List[any]],valor_para_balancear = 0) -> List[List[any]]:
  copy_list = matrix.copy()
  len_maximo = max([ len(v) for v in copy_list])
  for g in copy_list:
   if len(g) < len_maximo:
      diferencia = len_maximo - len(g)
      for _ in range(diferencia):
         g.append(valor_para_balancear)
  return copy_list

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

def element_in_matrix(elemento: any, matrix: List[List[any]]) -> List[Tuple[int,int]]:
  n = len(matrix)
  m = len(matrix[0])
  elements = []
  for i in range(n):
    for j in range(m):
      if matrix[i][j] == elemento:
        elements.append((i,j))
      else:
        continue
  return elements
      
def save_json(datos,file: str) -> None:
  with open(file,"w", encoding="utf-8") as sj:
    json.dump(datos, sj, indent=4, ensure_ascii=False)

def read_json(file: str):
  with open(file) as rj:
    datos = json.load(rj)
  return datos


def coeficiente(ind: List[List[int]], dep: List[int], grade : int=1) -> float:
  model = LinearRegression()
  poly = PolynomialFeatures(degree=grade)
  px = poly.fit_transform(np.array(ind).T)
  model.fit(px,np.array(dep))
  coef = model.score(px,dep)
  return coef
