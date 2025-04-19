

from Scripts.FeatureData.Encode import encode_sentences
from Scripts.FeatureData.Preprocess import preprocess_text

from transformers import AutoTokenizer, AutoModel

import joblib
import torch

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def Sentence(data_list):
    model_train = joblib.load('Model/model_lr.pkl')
    global count_true_news, count_fake_news
    sentence_preprocess = preprocess_text(data_list)
    sentence_vectorizer = encode_sentences(sentence_preprocess, tokenizer, model, device)
    sentence_pred = model_train.predict(sentence_vectorizer)
    sentence_probs = model_train.predict_proba(sentence_vectorizer)
    
    count_true_news = 0
    count_fake_news = 0

    for i in range(len(sentence_pred)):
        sentence_accuracy = max(sentence_probs[i]) * 100  # ความมั่นใจของโมเดล

        if sentence_pred[i] == 'ข่าวจริง' or sentence_pred[i] == 1:
            # if (len(data_list) <= 1): 
            #     print(f'ข่าวที่ {i + 1}: {data_list[i]} || ข่าวจริง: {sentence_accuracy:.2f}%')
            count_true_news += 1
        else:
            # if (len(data_list) <= 1): 
            #     print(f'ข่าวที่ {i + 1}: {data_list[i]} || ข่าวปลอม:{sentence_accuracy:.2f}%')
            count_fake_news += 1

    labels = ''
    if (sentence_pred[i] == 0): 
        labels = 'ข่าวปลอม'
    else: 
        labels = 'ข่าวจริง'
    
    return f'ข่าว: {data_list[0]} || {labels}: {sentence_accuracy:.2f}%'