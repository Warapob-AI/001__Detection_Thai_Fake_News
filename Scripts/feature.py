from .data_cleaning import Lemmatization, Remove_Special_character, Tokenization, Lowercase
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, CamembertTokenizer

import numpy as np
import joblib
import torch 

# def Import_Data(list_fake_news, list_true_news):
#     list_fake_news = list_fake_news
#     list_true_news = list_true_news

#     sentences = list_fake_news + list_true_news
#     labels = [0] * len(list_fake_news) + [1] * len(list_true_news)

#     x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size = 0.1, random_state=42)

#     return x_train, x_test, y_train, y_test

def preprocess_text(text):
    text = Lemmatization(text)
    text = Remove_Special_character(text)
    text = Tokenization(text)
    text = Lowercase(text)
    return text

def encode_sentences(sentences, batch_size=16):
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    count = 0
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch = preprocess_text(batch)
        # print(batch)
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embeddings)
        count += 1
        # print(count)

    return np.vstack(all_embeddings)

def Text_To_BERT(fake_news, true_news): 
    fake_news_features = encode_sentences(fake_news)
    print('Test fake_news_features Success ✅')
    true_news_features = encode_sentences(true_news)
    print('Test true_news_features Success ✅')

    return fake_news_features, true_news_features

def Save_Vectorizer(fake_news, true_news):
    joblib.dump(fake_news, 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/fake_news_features.pkl')
    joblib.dump(true_news, 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/true_news_features.pkl')

def Load_Vectorizer(fake_news, true_news): 
    # โหลดเวกเตอร์จากไฟล์
    fake_news_features = joblib.load(fake_news)
    true_news_features = joblib.load(true_news)

    return fake_news_features, true_news_features