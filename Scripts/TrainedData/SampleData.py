from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib


def SampleData(fake_news_features, true_news_features):
    if isinstance(fake_news_features, np.ndarray) and isinstance(true_news_features, np.ndarray):
        X = np.concatenate((fake_news_features, true_news_features), axis=0)
    elif isinstance(fake_news_features, list) and isinstance(true_news_features, list):
        X = fake_news_features + true_news_features
    else:
        raise ValueError("ข้อมูลในไฟล์ไม่ตรงกับรูปแบบที่คาดหวัง (numpy array หรือ list)")

    y = [0] * len(fake_news_features) + [1] * len(true_news_features)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    return x_train, x_test, y_train, y_test
