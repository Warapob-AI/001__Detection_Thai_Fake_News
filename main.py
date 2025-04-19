<<<<<<< HEAD
from Scripts.CleanedData.LoadData import Load_data
from Scripts.CleanedData.Lemmatization import Lemmatization
from Scripts.CleanedData.Tokenization import Tokenization
from Scripts.CleanedData.Special import Remove_Special_character
from Scripts.CleanedData.Lowercase import Lowercase

from Scripts.FeatureData.Encode import encode_by_label
from Scripts.FeatureData.SaveVector import Save_Vectorizer
from Scripts.FeatureData.LoadVector import Load_Vectorizer

from Scripts.TrainedData.SampleData import SampleData
from Scripts.TrainedData.LogisticRegression import Logistic_Regression
from Scripts.TrainedData.SVM import Support_Vector_Machine

from Scripts.EvaluationData.Classification import Classification_Report
from Scripts.EvaluationData.Sentence import Sentence

from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from fastapi import FastAPI
from mangum import Mangum

import pandas as pd
import numpy as np
import uvicorn
import torch 
import os 

if not os.path.exists('model/fakenews_features.pkl') or not os.path.exists('model/truenews_features.pkl'):
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    df_fake = Load_data('Dataset/fake_news.txt', 'fakenews')
    df_true = Load_data('Dataset/true_news.txt', 'truenews')

    # # รวมข้อมูลสอง DataFrame เข้าด้วยกัน
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df['cleaned_text'] = df['text'].apply(Lemmatization)
    df['cleaned_text'] = df['cleaned_text'].apply(Tokenization)
    df['cleaned_text'] = df['cleaned_text'].apply(Remove_Special_character)
    df['cleaned_text'] = df['cleaned_text'].apply(Lowercase)

    label_features = encode_by_label(df, tokenizer, model, device)

    # แยก features ตาม label
    fake_news_features = label_features['fakenews']
    true_news_features = label_features['truenews']

    Save_Vectorizer(fake_news_features, true_news_features)

if not os.path.exists('model/model_lr.pkl'):
    fakenews_data = Load_Vectorizer('Model/fakenews_features.pkl')
    truenews_data = Load_Vectorizer('Model/truenews_features.pkl')

    x_train, x_test, y_train, y_test = SampleData(fakenews_data, truenews_data)
    y_pred, name = Support_Vector_Machine(x_train, x_test, y_train, y_test)
    Classification_Report(name, y_test, y_pred)

app = FastAPI()
mangum = Mangum(app)

# สร้าง API Input สำหรับข้อความที่ต้องการประมวลผล
class TextInput(BaseModel):
    text: str

# สร้าง endpoint สำหรับรับข้อความและประมวลผล
@app.post("/")
async def predict(text_input: TextInput):
    # รับข้อความจากผู้ใช้
    text = text_input.text
    
    # ใช้ฟังก์ชัน Sentence เพื่อแปลงข้อความ
    result = Sentence([text])
    
    # ส่งผลลัพธ์ที่แปลงเป็นเวกเตอร์กลับไป
    return {"result": result}
=======
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
>>>>>>> 39a21087d3f37c92b3e469d4bb4ee1430b13e831
