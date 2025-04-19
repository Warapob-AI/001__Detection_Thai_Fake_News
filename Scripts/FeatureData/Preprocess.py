from Scripts.CleanedData.Lemmatization import Lemmatization
from Scripts.CleanedData.Tokenization import Tokenization
from Scripts.CleanedData.Special import Remove_Special_character
from Scripts.CleanedData.Lowercase import Lowercase

def preprocess_text(data_list):
    processed_list = []

    for text in data_list:
        cleaned_text = Lemmatization(text)  # หรือสามารถเปลี่ยนให้ตรงกับฟังก์ชันที่ใช้
        cleaned_text = Tokenization(cleaned_text)
        cleaned_text = Remove_Special_character(cleaned_text)
        cleaned_text = Lowercase(cleaned_text)

        processed_list.append(cleaned_text)

    return processed_list

