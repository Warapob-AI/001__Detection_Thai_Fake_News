import joblib 

def Save_Vectorizer(fake_news, true_news):
    joblib.dump(fake_news, 'Model/fakenews_features.pkl')
    joblib.dump(true_news, 'Model/truenews_features.pkl')