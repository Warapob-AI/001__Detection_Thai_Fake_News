import joblib
from sklearn.linear_model import LogisticRegression

def Logistic_Regression(x_train, x_test, y_train, y_test):
    name = 'Logistic_Regression'
    model = LogisticRegression(max_iter=2000, C=1, tol=0.001)
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)  

    joblib.dump(model, 'Model/model_lr.pkl')

    return y_pred, name