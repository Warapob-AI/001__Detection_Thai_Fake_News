from matplotlib import pyplot as plt
from .feature import encode_sentences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def Logistic_Regression(fake_news_features, true_news_features):
    if isinstance(fake_news_features, np.ndarray) and isinstance(true_news_features, np.ndarray):
        X = np.concatenate((fake_news_features, true_news_features), axis=0)
    elif isinstance(fake_news_features, list) and isinstance(true_news_features, list):
        X = fake_news_features + true_news_features
    else:
        raise ValueError("ข้อมูลในไฟล์ไม่ตรงกับรูปแบบที่คาดหวัง (numpy array หรือ list)")

    y = [0] * len(fake_news_features) + [1] * len(true_news_features)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=94)

    name = 'Logistic_Regression'
    model = LogisticRegression(max_iter=2000, C=1, tol=0.001)
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)  

    joblib.dump(model, 'my-portfolio-web/001__Detection_Thai_Fake_News/Model/model_lr.pkl')

    return x_train, x_test, y_train, y_test, y_pred, name

def Classification_Report(name, y_test, y_pred): 
    accuracy = accuracy_score(y_test, y_pred)
    print(f'ความแม่นยำของโมเดล {name} : {accuracy * 100:.2f}')

    classification = classification_report(y_test, y_pred, target_names=['Fake News', 'True News'])
    print(classification)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f'True Positive (TP): {tp}')
        print(f'False Positive (FP): {fp}')
        print(f'True Negative (TN): {tn}')
        print(f'False Negative (FN): {fn}')
    else:
        print("Confusion matrix is not binary. Please check the number of classes.")

def Sentence(data_list):
    model_train = joblib.load('Model/model_lr.pkl')
    global count_true_news, count_fake_news
    sentence_vectorizer = encode_sentences(data_list)
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

    return f'{labels} ({sentence_accuracy:.2f}%)'

def Learning_Curve(x_train, x_test, y_train, y_test):
    model = joblib.load('001__Detection_Thai_Fake_News/Model/model.lr.pkl')

    # คำนวณ accuracy สำหรับ training และ test sets
    train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    # แสดงผล accuracy
    print(f'Train Accuracy: {train_accuracy * 100:.2f}% | Test Accuracy: {test_accuracy * 100:.2f}%')

    # สร้างกราฟเส้น (Line chart)
    plt.figure(figsize=(10, 6))
    plt.plot(['Train'], [train_accuracy], label='Train Accuracy', marker='o', color='blue')
    plt.plot(['Test'], [test_accuracy], label='Test Accuracy', marker='o', color='orange')

    plt.title('Train vs Test Accuracy')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim([0, 1])  # Set the y-axis limit to [0, 1] for clarity
    plt.grid(True)
    plt.show()
    
def Test_Fake_News(data_list): 
    Sentence(data_list)
    print(f'ในการทดสอบข่าวปลอมทั้งหมด {len(data_list)} ข่าว, โมเดลทำนายผิดว่าเป็นข่าวจริง : {count_true_news} ข่าว')

def Test_True_News(data_list): 
    Sentence(data_list)
    print(f'ในการทดสอบข่าวจริงทั้งหมด {len(data_list)} ข่าว, โมเดลทำนายผิดว่าเป็นข่าวปลอม : {count_fake_news} ข่าว')

def Test_Logistic_Regression(fake_news_features, true_news_features, i):
    if isinstance(fake_news_features, np.ndarray) and isinstance(true_news_features, np.ndarray):
        X = np.concatenate((fake_news_features, true_news_features), axis=0)
    elif isinstance(fake_news_features, list) and isinstance(true_news_features, list):
        X = fake_news_features + true_news_features
    else:
        raise ValueError("ข้อมูลในไฟล์ไม่ตรงกับรูปแบบที่คาดหวัง (numpy array หรือ list)")

    y = [0] * len(fake_news_features) + [1] * len(true_news_features)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=i)

    name = 'Logistic_Regression'
    model = LogisticRegression(max_iter=2000, C=1, tol=0.001, random_state=i)
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)  

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random State ที่ {i} | มีความแม่นยำของโมเดล {name} : {accuracy * 100:.2f}')
    print(classification_report(y_test, y_pred, target_names=['Fake News', 'True News']))

def Test_Random_State(fake_news_features, true_news_features): 
    i = 0

    while i <= 100: 
        Test_Logistic_Regression(fake_news_features, true_news_features, i)
        i += 1
