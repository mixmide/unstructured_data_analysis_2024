def is_end_of_file(file):
    current_position = file.tell()
    file.seek(0, 2)  # Перемещаемся в конец файла
    end_position = file.tell()
    file.seek(current_position)  # Возвращаемся на исходную позицию
    return current_position == end_position

def print_sentences_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            sentences = []
            current_sentence = ''
            first_kav = 0
            prev_kav = 0
            for char in text:
                current_sentence += char
                if char in {'.', '!', '?', '"'}:
                    if char == '"':
                        if first_kav == 1:
                            prev_kav = 1
                            sentences.append(current_sentence.strip())
                            current_sentence = ''
                            first_kav = 0
                        else:
                            first_kav = 1
                        continue
                    if prev_kav == 1:
                        sentences[-1] += char
                        prev_kav = 0
                        continue
                    if first_kav == 1:
                        continue
                    sentences.append(current_sentence.strip())
                    current_sentence = ''
            if current_sentence and not is_end_of_file(file):
                sentences.append(current_sentence.strip())
            number = 1
            for sentence in sentences:
                print(number, "-", sentence)
                number = number + 1

    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


file_path = 'test_rus.txt'
print_sentences_from_file(file_path)

'''
Программа может некорректно работать при сокращении слов с использованием точки и при прямой речи (то есть с кавычками).
'''