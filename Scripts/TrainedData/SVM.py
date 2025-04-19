import joblib
from sklearn.svm import SVC

def Support_Vector_Machine(x_train, x_test, y_train, y_test):
    name = 'Support_Vector_Machine'
    model = SVC(kernel='linear', C=1.0)  # ใช้ linear kernel, สามารถเปลี่ยนเป็น 'rbf', 'poly' ได้
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    joblib.dump(model, 'Model/model_svm.pkl')

    return y_pred, name
