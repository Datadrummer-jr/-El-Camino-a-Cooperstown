
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
    sumas.append(float(sum(traspuesta[i])))
  maximo = max(sumas)
  return sumas.index(maximo)

def min_valor(matrix: List[List[int]]) -> List[List[int]]:
  matrix = np.array(matrix)
  traspuesta = matrix.T
  sumas = []
  for i in range(len(traspuesta)):
    sumas.append(float(sum(traspuesta[i])))
  maximo = min(sumas)
  return sumas.index(maximo)





