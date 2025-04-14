import gradio as gr
import os

from Scripts.data_cleaning import Load_data
from Scripts.feature import Text_To_BERT, Save_Vectorizer, Load_Vectorizer
from Scripts.train_model import Logistic_Regression, Classification_Report, Sentence, Test_Random_State

path_fake_news = 'Model/fake_news_features.pkl'
path_true_news = 'Model/true_news_features.pkl'
path_model = 'Model/model_lr.pkl'

def predict_text(input_data):
    sentence = input_data
    result = Sentence([sentence])
    return result

def gradio_interface(input_data):
    if os.path.exists(path_model):
        return predict_text(input_data)

    elif os.path.exists(path_fake_news) and os.path.exists(path_true_news):
        fake_news_features, true_news_features = Load_Vectorizer(path_fake_news, path_true_news)
        x_train, x_test, y_train, y_test, y_pred, name = Logistic_Regression(fake_news_features, true_news_features)
        Classification_Report(name, y_test, y_pred)
        return f"Model Trained and Report Generated for {name}"

    else: 
        fake_news = Load_data('Dataset/fake_news.txt')
        true_news  = Load_data('Dataset/true_news.txt')
        fake_news_vec, true_news_vec = Text_To_BERT(fake_news, true_news)
        Save_Vectorizer(fake_news_vec, true_news_vec)
        return "Vectorizer Saved and Data Preprocessed"

# Gradio UI
interface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", live=True)

if __name__ == "__main__":
    interface.launch()
