# программа для печати слов вместе с их словоформами и их подсчета (для русского языка)

import pymorphy2

def print_word_forms(word):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(word)[0]
    forms = p.lexeme
    print("Слово:", word)
    print("Количество форм:", len(forms))
    for form in forms:
        print(form.word)
    print()

def find_max_forms_word(words):
    morph = pymorphy2.MorphAnalyzer()
    max_forms = 0
    max_word = ""
    for word in words:
        p = morph.parse(word)[0]
        forms = p.lexeme
        if len(forms) > max_forms:
            max_forms = len(forms)
            max_word = word
    return max_word, max_forms

# печать всех слов с их словоформами
words = ["читать", "слушать", "мир", "красивый"]         # здесь анализируемые слова
for word in words:
    print_word_forms(word)

# печать подсчета слов
max_word, max_forms = find_max_forms_word(words)
print("Слово с максимальным количеством словоформ: ", max_word)
print("Количество таких словоформ: ", max_forms)
