import joblib
import sys
import os 
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer
from transformers import CamembertTokenizer

sys.path.append(os.path.abspath('001__Detection_Thai_Fake_News/Scripts'))

from Scripts.data_cleaning import Load_data
from Scripts.feature import Text_To_BERT, Save_Vectorizer, Load_Vectorizer
from Scripts.train_model import Learning_Curve, Logistic_Regression, Classification_Report, Sentence, Test_Random_State

path_fake_news = 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/fake_news_features.pkl'
path_true_news = 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/true_news_features.pkl'
path_model = 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/model_lr.pkl'

if __name__ == '__main__':
    if (os.path.exists(path_fake_news) and os.path.exists(path_true_news)):
        fake_news_features, true_news_features = Load_Vectorizer(path_fake_news, path_true_news)
        x_train, x_test, y_train, y_test, y_pred, name = Logistic_Regression(fake_news_features, true_news_features)
        Classification_Report(name, y_test, y_pred)

