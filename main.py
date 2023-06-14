from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Definisikan BaseModel untuk input data pada endpoint POST
class DataInput(BaseModel):
    data: int

# mengImport model TensorFlow ,  menggunakan model linear
model = tf.keras.models.load_model('./linear.h5')

@app.get("/")
def hello():
    return {"message": "FastAPI CloudRun TensorFlow Deployment"}

# Contoh endpoint GET untuk melakukan prediksi dengan model
@app.get("/predict")
def predict():
    prediction = model.predict([[10.0]])
    result = prediction.item()
    return {"result": result}

# Contoh endpoint POST untuk melakukan prediksi dengan model
@app.post("/predict")
def predict(data: DataInput):
    prediction = model.predict([[data.data]])
    result = prediction.item()
    return {"result": result}
