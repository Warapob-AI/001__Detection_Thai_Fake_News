import joblib
import sys
import os 

sys.path.append(os.path.abspath('001__Detection_Thai_Fake_News/Scripts'))

from Scripts.data_cleaning import Load_data
from Scripts.feature_ import Text_To_BERT, Save_Vectorizer, Load_Vectorizer
from Scripts.train_model import Logistic_Regression, Classification_Report, Sentence, Test_Random_State

path_fake_news = '001__Detection_Thai_Fake_News/Model/fake_news_features.pkl'
path_true_news = '001__Detection_Thai_Fake_News/Model/true_news_features.pkl'

if __name__ == '__main__':
    if (os.path.exists(path_fake_news) and os.path.exists(path_true_news)):
        fake_news_features, true_news_features = Load_Vectorizer(path_fake_news, path_true_news)
        name, y_pred, y_test = Logistic_Regression(fake_news_features, true_news_features)
        Classification_Report(name, y_test, y_pred)

        sentence = str(input('\nEnter Sentence : '))
        result = Sentence([sentence])
        print(result)
        # Test_Random_State(fake_news_features, true_news_features)
    else:
        list_fake_news = Load_data('001__Detection_Thai_Fake_News/Dataset/fake_news.txt')
        list_true_news = Load_data('001__Detection_Thai_Fake_News/Dataset/true_news.txt')

        fake_news_features, true_news_features = Text_To_BERT(list_fake_news, list_true_news)
        Save_Vectorizer(fake_news_features, true_news_features)
        
