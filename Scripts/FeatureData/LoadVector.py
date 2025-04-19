import joblib 

def Load_Vectorizer(datanews): 
    datanews = joblib.load(datanews)

    return datanews