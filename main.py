import sys
import os 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from Scripts.train_model import Sentence

path_model = 'Model/model_lr.pkl'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

@app.post("/fakenews")
def predict_text(input_data: InputText):
    sentence = input_data.text
    # ดึงผลลัพธ์จากโมเดลจริง ๆ ที่คุณเขียนไว้
    result = Sentence([sentence])
    print(result)
    return {"result": result}

