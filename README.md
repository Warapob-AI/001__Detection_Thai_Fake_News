# 📌 Fake News Detection
[![fefwefew.png](https://i.postimg.cc/x1dbV0P1/fefwefew.png)](https://postimg.cc/svt2pC8k)
## 🔍 Overview
โครงการนี้เป็นระบบ **ตรวจสอบความน่าเชื่อถือของข่าว** โดยใช้ **Machine Learning** และ **Natural Language Processing (NLP)** เพื่อตรวจจับข่าวปลอมจากเนื้อหาข่าวแบบข้อความ

## ⚙️ การทำงานของระบบ
1. **โหลดและประมวลผลข้อมูล**
   - ดึงข้อมูลจากแหล่งข่าว
   - แปลงข้อความเป็นเวกเตอร์ด้วย **DEBERT หรือ BERT Wangchan**
   - ใช้ **Logistic Regression** เป็นโมเดลหลักในการจำแนกข่าวจริง-ข่าวปลอม

2. **Feature Engineering**
   - ตรวจจับหัวข้อข่าว
     
3. **การพยากรณ์ (Prediction)**
   - เมื่อป้อน URL หรือเนื้อหาข่าว ระบบจะทำนายว่าข่าวนั้น **"น่าเชื่อถือ" หรือ "ข่าวปลอม"**

## 🚀 เทคโนโลยีที่ใช้
- **Python**
- **Logistic Regression**
- **DEBERT / BERT**
- **Machine Learning**
- **NLP**

## 📦 การติดตั้งและใช้งาน
### 1️⃣ ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ รันโค้ด main.py 
```bash
sentence = str(input('\nEnter Sentence : '))
result = Sentence([sentence])

# ใส่ข้อความ input ลงใน sentence
```
