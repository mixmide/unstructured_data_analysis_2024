import requests
from bs4 import BeautifulSoup

def count_string_occurrences(url, search_string):
    try:
        # Отправляем GET-запрос к указанному URL
        response = requests.get(url)
        response.raise_for_status()  # Проверяем наличие ошибок

        # Получаем содержимое страницы
        page_content = response.text

        # Используем BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(page_content, 'html.parser')

        # Получаем текст страницы
        page_text = soup.get_text()

        # Считаем количество вхождений строки
        occurrences = page_text.count(search_string)

        return occurrences
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return 0

def main():
    # Вводим название сайта и строку
    url = input("Введите URL сайта: ")
    search_string = input("Введите строку для поиска: ")

    # Считаем количество вхождений строки на сайте
    occurrences = count_string_occurrences(url, search_string)

    # Выводим результат
    print(f"Строка '{search_string}' встречается на сайте {occurrences} раз(а).")

if __name__ == "__main__":
    main()
