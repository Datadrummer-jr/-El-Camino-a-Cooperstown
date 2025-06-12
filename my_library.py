
def you_type(cadena: str):
  n = len(cadena)
  caracteres = [l for l in cadena]
  if caracteres[0] == "." and cadena[1:n].isdigit():
    return float(cadena)
  if cadena.isdigit():
    return int(cadena)
  else:
    for i in cadena:
      if "." in caracteres and i.isdigit():
        return float(cadena)
      else:
        return str(cadena)
      