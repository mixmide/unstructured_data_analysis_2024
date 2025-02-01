#ДАННАЯ ПРОГРАММА СОЗДАЕТ СТАРТОВЫЙ ДАТАСЕТ, БЕЗ АУГМЕНТАЦИИ ___________________________________________________________
#импорт библиотек
import os
import string
import random
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

# инициализация наших параметров
DATASET_PATH = "start_dataset" #здесь храним стартовый датасет
IMG_SIZE = (128, 128) #размер изображения с образцом буквы
FONT_SIZE = 100 #размер шрифта
NUM_SAMPLES = 100  #кол-во образцов для каждой буквы (однако после ручной проверки станет 90)

#получаем список установленных на компе шрифтов формата ttf
def get_system_fonts():
    return [font for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf')]

#создаем изображение с символом и сохраняем его
def create_image(char, font_path, save_path):
    image = Image.new("L", IMG_SIZE, color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, FONT_SIZE)
    #центрирование текста
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((IMG_SIZE[0] - text_width) // 2 - bbox[0],
                (IMG_SIZE[1] - text_height) // 2 - bbox[1])
    draw.text(position, char, font=font, fill="black")
    image.save(save_path)

#эта функция создает датасет из заглавных букв
def create_dataset():
    os.makedirs(DATASET_PATH, exist_ok=True)
    fonts = get_system_fonts()
    for char in string.ascii_uppercase:
        char_dir = os.path.join(DATASET_PATH, char)
        os.makedirs(char_dir, exist_ok=True)
        count = 0
        while count < NUM_SAMPLES:
            font_path = random.choice(fonts)
            save_path = os.path.join(char_dir, f"{char}_{count}.png")
            try:
                create_image(char, font_path, save_path)
                count += 1
            except Exception as e:
                print(f"Ошибка с шрифтом {font_path} для буквы '{char}': {e}")

    print(f"Датасет создан в папке {DATASET_PATH}")

if __name__ == "__main__":
    create_dataset()

#После выполнения этой программы будет создана папка start_dataset, в которой для каждой буквы будет создана папка, хранящая 100
#образцов буквы. Однако она может содержать некорректные символы (прямоугольники, белый фон и тп.), поэтому данный датасет нужно
#вручную сократить до 90 элементов, удалив из него некорректные символы. Устранить данную проблему на программном уровне не удалось.