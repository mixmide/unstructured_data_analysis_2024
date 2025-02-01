# рекурентная сеть прогноз букв
import os
import re
import numpy as np
from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# варианты для экспериментов
variants = {
    "letters_only": "А-яё ",
    "letters_spaces_dashes": "А-яё\- ",
    "letters_spaces_dashes_dots": "А-яё\-\. ",
    "full_set": "А-яё\-\. ,"
}

# функция предобработки текста
def preprocess_text(file_path, allowed_chars):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\ufeff', '')  #убираем невидимый символ
        text = re.sub(fr'[^{allowed_chars}]', ' ', text)  #убираем лишние символы
        text = re.sub(r' +', ' ', text)  #убираем лишние пробелы
        text = text.lower()
    return text

#подготовка данных
file_path = 'cat.txt'
preprocessed_texts = {
    name: preprocess_text(file_path, allowed_chars)
    for name, allowed_chars in variants.items()
}

# здесь обучение и предсказание
inp_chars = 3
num_epochs = 50
def train_and_evaluate(text, variant_name):
    num_characters = len(set(text))  # размер словаря
    tokenizer = Tokenizer(num_words=num_characters, char_level=True)
    tokenizer.fit_on_texts(text)
    data = tokenizer.texts_to_matrix(text)
    n = data.shape[0] - inp_chars
    X = np.array([data[i:i+inp_chars, :] for i in range(n)])
    Y = data[inp_chars:]
    #модель
    model = Sequential()
    model.add(Input((inp_chars, num_characters)))
    model.add(SimpleRNN(500, activation='tanh'))
    model.add(Dense(num_characters, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
    model.fit(X, Y, batch_size=64, epochs=num_epochs, verbose=False)

    #генерация текста
    def build_phrase(inp_str, str_len=50):
        for _ in range(str_len):
            x = np.array([
                tokenizer.texts_to_matrix(inp_str[i:i+inp_chars])
                for i in range(len(inp_str) - inp_chars + 1)
            ])
            pred = model.predict(x, verbose=False)[-1]
            next_char = tokenizer.index_word[pred.argmax(axis=0)]
            inp_str += next_char
        return inp_str

    #оценка
    test_seed = "кот"
    generated = build_phrase(test_seed)
    #BLEU-метрика (доля совпадающих символов)
    def simple_accuracy(true_text, predicted_text):
        matches = sum(1 for t, p in zip(true_text, predicted_text) if t == p)
        return matches / len(true_text)
    #доля совпадающих n-грамм
    def ngram_accuracy(true_text, predicted_text, n=3):
        true_ngrams = [true_text[i:i+n] for i in range(len(true_text)-n+1)]
        pred_ngrams = [predicted_text[i:i+n] for i in range(len(predicted_text)-n+1)]
        matches = sum(1 for ngram in pred_ngrams if ngram in true_ngrams)
        return matches / len(true_ngrams)

    acc = simple_accuracy(text[:len(generated)], generated[:len(text)])
    ngram_acc = ngram_accuracy(text[:len(generated)], generated[:len(text)])
    print(f"\n=== Variant: {variant_name} ===")
    print(f"Generated: {generated[:100]}")
    print(f"Simple Accuracy: {acc:.4f}")
    print(f"3-Gram Accuracy: {ngram_acc:.4f}")
    return {"Simple Accuracy": acc, "3-Gram Accuracy": ngram_acc}

#запуск
results = {}
for variant_name, text_variant in preprocessed_texts.items():
    results[variant_name] = train_and_evaluate(text_variant, variant_name)

print("\n=== Summary of Results ===")
for variant, metrics in results.items():
    print(f"{variant}: Simple Accuracy = {metrics['Simple Accuracy']:.4f}, 3-Gram Accuracy = {metrics['3-Gram Accuracy']:.4f}")