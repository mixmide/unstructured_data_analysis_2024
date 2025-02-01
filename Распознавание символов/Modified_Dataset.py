#ДАННАЯ ПРОГРАММА СОЗДАЕТ ИТОГОВЫЙ ДАТАСЕТ, С АУГМЕНТАЦИЕЙ ___________________________________________________________
#импорт библиотек
import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import numpy as np

# инициализация наших параметров
INPUT_DATASET_PATH = "start_dataset"  #путь к исходному датасету
OUTPUT_DATASET_PATH = "modified_dataset"  #путь для сохранения модифицированного датасета
IMG_SIZE = (128, 128)  #размер изображений с образцами букв
NUM_AUGMENTED_SAMPLES = 270  #кол-во аугментированных (!) изображений для каждой буквы
RARE_AUGMENTATION_PROB = 0.1  #вероятность редких эффектов (царапины, размытие) на фото
NOISE_PROB = 0.15  #вероятность добавления шума
INVERSION_PROB = 0.15  #вероятность инверсии изображения, т.е. светлые буквы на темном фоне (все-таки светлые изображения у нас чаще)

#функция добавления царапин
def add_scratches(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(random.randint(1, 2)):  # Редко и минимально
        x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
        x2, y2 = random.randint(0, width - 1), random.randint(0, height - 1)
        draw.line([(x1, y1), (x2, y2)], fill=random.randint(200, 255), width=1)
    return image

#функция создания случайного фона
def generate_background(image, is_light_text):
    if is_light_text:
        #темный фон, если текст светлый
        bg_color = random.randint(30, 60)
    else:
        #светлый фон для темного текста
        bg_color = random.randint(200, 255)
    #создаем новый фон и объединяем его с изображением буквы
    background = Image.new("L", IMG_SIZE, bg_color)
    result_image = Image.composite(image, background, image)
    return result_image

#функция создания шума
def add_noise(image_array):
    noise = np.random.normal(0, 10, image_array.shape).astype('int16')
    noisy_image = np.clip(image_array + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_image)

#функция для аугментации изображения с буквой
def augment_image(image, is_light_text):
    #добавляем фон
    image = generate_background(image, is_light_text)
    #преобразование в массив
    img_array = np.array(image)
    #случайное добавление шума
    if random.random() < NOISE_PROB:
        image = add_noise(img_array)
    #"редкие" аугментации (царапины/размытие)
    if random.random() < RARE_AUGMENTATION_PROB:
        if random.random() < 0.5:  #размытие
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        else:  #царапины
            image = add_scratches(image)
    #случайная инверсия
    if random.random() < INVERSION_PROB:
        image = ImageOps.invert(image)
    return image

#функция создания модифицированного датасета
def create_modified_dataset():
    os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)
    for letter in os.listdir(INPUT_DATASET_PATH):
        letter_path = os.path.join(INPUT_DATASET_PATH, letter)
        if os.path.isdir(letter_path):
            output_letter_path = os.path.join(OUTPUT_DATASET_PATH, letter)
            os.makedirs(output_letter_path, exist_ok=True)
            #копирование оригинальных изображений
            original_images = sorted(os.listdir(letter_path))[:90]
            for idx, img_name in enumerate(original_images):
                img_path = os.path.join(letter_path, img_name)
                image = Image.open(img_path).convert("L").resize(IMG_SIZE)
                #проверяем, светлая ли буква
                is_light_text = np.array(image).max() > 128
                #добавляем правильный фон
                image = generate_background(image, is_light_text)
                image.save(os.path.join(output_letter_path, f"{idx}.png"))
            #здесь у нас создание аугментированных изображений
            count = 90
            while count < 360:
                img_name = random.choice(original_images)
                img_path = os.path.join(letter_path, img_name)
                image = Image.open(img_path).convert("L").resize(IMG_SIZE)
                #проверка на светлый текст
                is_light_text = np.array(image).max() > 128
                #аугментация
                augmented_image = augment_image(image, is_light_text)
                #сохранение
                augmented_image.save(os.path.join(output_letter_path, f"{count}.png"))
                count += 1
    print(f"Модифицированный датасет создан в папке {OUTPUT_DATASET_PATH}")

if __name__ == "__main__":
    create_modified_dataset()