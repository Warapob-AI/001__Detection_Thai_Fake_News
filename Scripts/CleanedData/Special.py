import re 

def Remove_Special_character(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9ก-ฮะ-์ํา]+', '', text)
    return cleaned_text

