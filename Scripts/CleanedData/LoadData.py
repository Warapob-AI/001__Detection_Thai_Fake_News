import pandas as pd 
import codecs

def Load_data(filename, label):
    with codecs.open(filename, 'r', 'utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return pd.DataFrame({'text': lines, 'cleaned_text': '', 'label': label})

df_fake = Load_data('Dataset/fake_news.txt', 'fakenews')
df_true = Load_data('Dataset/fake_news.txt', 'truenews')

# รวมข้อมูลสอง DataFrame เข้าด้วยกัน
df = pd.concat([df_fake, df_true], ignore_index=True)

print(df.head(10))