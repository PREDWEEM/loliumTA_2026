from fastapi import FastAPI
from predweem_core import ejecutar_predweem

app = FastAPI()

@app.post("/simular")
def simular():
    resultado = ejecutar_predweem(...)
    return resultado
