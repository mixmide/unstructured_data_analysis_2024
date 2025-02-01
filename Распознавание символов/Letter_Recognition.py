#ПРОГРАММА РАСПОЗНАВАНИЯ СИМВОЛОВ ___________________________________________________________
#импорт необходимых библиотек
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout #для построения сверточной нейросети
from tensorflow.keras.preprocessing.image import ImageDataGenerator #обработка изображений
import matplotlib.pyplot as plt #для графиков

#датасеты
train_dir = 'modified_dataset'  #папка с обучающими изображениями
test_dir = 'alphabet_dataset_test'  #папка с тестовыми изображениями

#гиперпараметры
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 64
EPOCHS = 8

#подготовка данных для дополнительной аугментации (сдвиг в разные стороны и масштабирование) и определение части для валидации
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # 20% данных для валидации
)

#нормализация для тестовой выборки
test_datagen = ImageDataGenerator(rescale=1./255)

#наш генератор для обучения
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  #используем 80% данных для обучения
)

#наш генератор для тестирования
test_data = test_datagen.flow_from_directory(
    test_dir,  #папка с изображениями для теста
    target_size=(IMG_HEIGHT, IMG_WIDTH),  #изменение размера изображений
    batch_size=BATCH_SIZE,  
    class_mode='categorical'  #множественная классификация, используется one-hot encoding
)

#генератор для данных валидации
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  #используем 20% данных для валидации
)

#построение модели
model = keras.Sequential([
    #1й сверточный слой с 32 фильтрами размером 3x3 и функцией активации ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),  #макс-пулинг с размером окна 2x2
    #2й сверточный слой с 64 фильтрами и функцией активации ReLU
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  #макс-пулинг с размером окна 2x2
    #3й сверточный слой с 128 фильтрами и функцией активации ReLU
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  #макс-пулинг с размером окна 2x2

    Flatten(),  #преобразование в одномерный вектор для подачи на полносвязные слои

    #полносвязный слой с 128 нейронами и функцией активации ReLU
    Dense(128, activation='relu'),
    
    #слой Dropout для регуляризации
    Dropout(0.5),

    #выходной слой с числом нейронов = количеству классов и функцией активации softmax
    Dense(len(train_data.class_indices), activation='softmax')
])

#компиляция модели: выбор оптимизатора (Adam), функции потерь и метрик
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  #для многоклассовой классификации
              metrics=['accuracy'])  #оценка точности

#обучение модели
history = model.fit(
    train_data, 
    epochs=EPOCHS,
    validation_data=val_data,
    verbose=1  #вывод инф-ии об обучении на каждом шаге
)

#оценка модели на тестовой выборке
loss, accuracy = model.evaluate(test_data, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

#построение графиков потерь и точности
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#НИЖЕ - ТЕСТИРОВАНИЕ НА КОНКРЕТНОМ ИЗОБРАЖЕНИИ (при необходимости можно раскомментировать или скопировать в новую программу)

#from tensorflow.keras.preprocessing import image
#import numpy as np
#img_path = 'image_k.png'  #проверяемое изображение
#img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
#img_array = image.img_to_array(img) #преобразуем изображение в массив
#img_array = np.expand_dims(img_array, axis=0)  #добавляем размерность для батча
#img_array /= 255.0  #нормализуем
#predictions = model.predict(img_array) #предсказания модели

#max_probability = np.max(predictions)  #наибольшая вероятность
#predicted_class = np.argmax(predictions)  #индекс класса с наибольшей вероятностью
#threshold = 0.6  #если максимальная вероятность меньше этого порога, то считаем, что буквы нет (подобрано экспериментально)

#if max_probability < threshold: #итоговая проверка
    #print("На изображении нет буквы.")
#else:
    #predicted_label = list(train_data.class_indices.keys())[predicted_class]  # Метка буквы
    #print(f"Предсказанный класс: {predicted_label} с вероятностью {max_probability*100:.2f}%")