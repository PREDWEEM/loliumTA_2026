
from fastapi import FastAPI, UploadFile
import pandas as pd

from predweem_core import ejecutar_predweem

app = FastAPI(
    title="PREDWEEM API"
)

@app.get("/")
def root():
    return {
        "modelo": "PREDWEEM",
        "estado": "OK"
    }

@app.post("/simular")
async def simular(
    clima: UploadFile,
    campo: UploadFile
):

    df_clima = pd.read_excel(clima.file)
    df_campo = pd.read_excel(campo.file)

    resultado = ejecutar_predweem(
        df_clima,
        df_campo,
        {}
    )

    return resultado
