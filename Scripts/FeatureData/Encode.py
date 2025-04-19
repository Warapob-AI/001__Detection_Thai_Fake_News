import numpy as np
import torch 

def encode_by_label(df, tokenizer, model, device, text_column='cleaned_text', label_column='label'):
    label_features = {}
    unique_labels = df[label_column].unique()

    for label in unique_labels:
        subset = df[df[label_column] == label]
        texts = subset[text_column].tolist()
        embeddings = encode_sentences(texts, tokenizer, model, device)
        label_features[label] = embeddings

    return label_features

def encode_sentences(sentences, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size  # จำนวน batch ทั้งหมด

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embeddings)

        # ✅ แสดงว่าถึง batch ที่เท่าไหร่แล้ว
        print(f'Processed batch {i // batch_size + 1}/{total_batches}')

    return np.vstack(all_embeddings)
