FROM python:3.10-slim

WORKDIR /app

# Optimizar caché de capas instalando requerimientos primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y los archivos binarios (.npy y .pkl) al contenedor
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
