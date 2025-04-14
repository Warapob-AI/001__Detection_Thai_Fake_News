import sys
import os 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum  # เพิ่มตัวนี้
from Scripts.train_model import Sentence

# import os

# from Scripts.data_cleaning import Load_data
# from Scripts.feature import Text_To_BERT, Save_Vectorizer, Load_Vectorizer
# from Scripts.train_model import Logistic_Regression, Classification_Report, Sentence, Test_Random_State

# path_fake_news = 'Model/fake_news_features.pkl'
# path_true_news = 'Model/true_news_features.pkl'
# path_model = 'Model/model_lr.pkl'

# def predict_text(input_data):
#     sentence = input_data
#     result = Sentence([sentence])
#     return result

# def interface(input_data):
#     if os.path.exists(path_model):
#         return predict_text(input_data)

#     elif os.path.exists(path_fake_news) and os.path.exists(path_true_news):
#         fake_news_features, true_news_features = Load_Vectorizer(path_fake_news, path_true_news)
#         x_train, x_test, y_train, y_test, y_pred, name = Logistic_Regression(fake_news_features, true_news_features)
#         Classification_Report(name, y_test, y_pred)
#         return f"Model Trained and Report Generated for {name}"

#     else: 
#         fake_news = Load_data('Dataset/fake_news.txt')
#         true_news  = Load_data('Dataset/true_news.txt')
#         fake_news_vec, true_news_vec = Text_To_BERT(fake_news, true_news)
#         Save_Vectorizer(fake_news_vec, true_news_vec)
#         return "Vectorizer Saved and Data Preprocessed"
    



path_model = 'Model/model_lr.pkl'

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รับข้อมูลจาก frontend
class InputText(BaseModel):
    text: str

@app.post("/")
def predict_text(input_data: InputText):
    sentence = input_data.text
    result = Sentence([sentence])
    print(result)
    return {"result": result}

# สำหรับ Vercel: ต้องมี handler แบบนี้
handler = Mangum(app)