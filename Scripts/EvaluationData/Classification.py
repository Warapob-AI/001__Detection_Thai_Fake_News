from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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