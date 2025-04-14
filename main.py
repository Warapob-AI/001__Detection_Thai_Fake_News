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

@app.post("/")
def predict_text(input_data: InputText):
    sentence = input_data.text
    result = Sentence([sentence])
    print(result)
    return {"result": result}

# # ✅ เพิ่มตรงนี้ เพื่อให้ Railway รู้วิธีรัน API
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080)
