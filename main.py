from tkinter import Tk, filedialog  # Импортируем базовый класс окна и диалог для выбора файлов
from tkinter import ttk  # Импортируем модуль для работы с виджетами
from PIL import ImageTk, Image  # Импортируем модули для работы с изображениями
import numpy as np  # Импортируем библиотеку для работы с массивами
import cv2  # Импортируем библиотеку для компьютерного зрения

# Создаем класс Root, который будет являться основным окном приложения
class Root(Tk):
    def __init__(self):  # Инициализация приложения
        super().__init__()  # Вызываем инициализацию родительского класса
        self.title("Image Colorization")  # Устанавливаем заголовок окна
        self.minsize(300, 500)  # Задаем минимальные размеры окна

        # Создаем рамки для группировки элементов интерфейса
        self.labelFrame = ttk.LabelFrame(self, text='Input Image')  # Рамка для кнопки выбора файла
        self.labelFrame.grid(column=0, row=1, padx=5, pady=5)  # Устанавливаем расположение рамки

        self.labelFrame1 = ttk.LabelFrame(self, text='Path')  # Рамка для отображения пути к файлу
        self.labelFrame1.grid(column=0, row=3, padx=5, pady=5)

        self.labelFrame2 = ttk.LabelFrame(self, text='Image')  # Рамка для отображения выбранного изображения
        self.labelFrame2.grid(column=0, row=4, padx=5, pady=5)

        self.labelFrame3 = ttk.LabelFrame(self, text='Run')  # Рамка для кнопки запуска программы
        self.labelFrame3.grid(column=0, row=5, padx=5, pady=5)

        self.button()  # Добавляем кнопки на созданные рамки

    def button(self):  # Метод для добавления кнопок
        # Кнопка для выбора файла
        ttk.Button(self.labelFrame, text='Browse File', width=50, command=self.fileDialog).grid(column=0, row=1)
        # Кнопка для запуска программы
        ttk.Button(self.labelFrame3, text='Run Program', width=50, command=self.RunPro).grid(column=0, row=1)

    def RunPro(self):  # Метод для запуска программы раскрашивания
        try:
            myimage = self.path  # Получаем путь к выбранному изображению

            print("[INFO] loading model...")  # Выводим сообщение о загрузке модели
            net = cv2.dnn.readNetFromCaffe(  # Загружаем модель для раскрашивания
                "models/colorization_deploy_v2.prototxt",
                "models/colorization_release_v2.caffemodel"
            )
            pts = np.load("models/pts_in_hull.npy")  # Загружаем массив с кластерными центрами

            # Получаем идентификаторы слоев модели
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2, 313, 1, 1)  # Преобразуем массив
            net.getLayer(class8).blobs = [pts.astype("float32")]  # Устанавливаем веса слоя class8
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]  # Устанавливаем веса слоя conv8

            image = cv2.imread(myimage)  # Читаем изображение
            scaled = image.astype("float32") / 255.0  # Масштабируем значения пикселей
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)  # Переводим изображение в цветовое пространство LAB

            resized = cv2.resize(lab, (224, 224))  # Изменяем размер изображения для модели
            L = cv2.split(resized)[0]  # Извлекаем канал L (яркость)
            L -= 50  # Нормализуем значения яркости

            print("[INFO] colorizing image...")  # Выводим сообщение о раскрашивании
            net.setInput(cv2.dnn.blobFromImage(L))  # Передаем яркость на вход модели
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # Получаем раскрашенные каналы a и b
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))  # Масштабируем каналы до исходного размера

            L = cv2.split(lab)[0]  # Повторно извлекаем яркость из исходного изображения
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)  # Объединяем каналы L, a и b
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)  # Преобразуем обратно в цветовое пространство BGR
            colorized = np.clip(colorized, 0, 1)  # Ограничиваем значения пикселей
            colorized = (255 * colorized).astype("uint8")  # Преобразуем значения пикселей в целые числа
            colorized = cv2.resize(colorized, (300, 350))  # Масштабируем изображение для отображения

            cv2.imshow("Colorized", colorized)  # Показываем раскрашенное изображение
            cv2.waitKey(0)  # Ожидаем нажатия клавиши
        except AttributeError:  # Обрабатываем ошибку, если файл не был выбран
            print("[ERROR] No file selected or invalid path")

    def fileDialog(self):  # Метод для выбора файла
        self.filename = filedialog.askopenfilename(  # Открываем диалог выбора файла
            initialdir='/', title='Select file',
            filetypes=(('JPEG files', '*.jpg'), ('All Files', '*.*'))
        )
        if self.filename:  # Если файл был выбран
            self.e1 = ttk.Entry(self.labelFrame1, width=50)  # Создаем текстовое поле для отображения пути
            self.e1.insert(0, self.filename)  # Вставляем путь в текстовое поле
            self.e1.grid(row=2, column=0, columnspan=50)  # Размещаем текстовое поле

            self.path = self.filename.replace('/', '\\')  # Преобразуем путь в формат Windows
            print(self.path)  # Выводим путь в консоль

            try:
                im = Image.open(self.path)  # Открываем изображение
                resized = im.resize((300, 300))  # Изменяем размер изображения для предпросмотра
                tkimage = ImageTk.PhotoImage(resized)  # Конвертируем изображение для tkinter
                myvar = ttk.Label(self.labelFrame2, image=tkimage)  # Создаем виджет для отображения изображения
                myvar.image = tkimage  # Сохраняем ссылку на изображение
                myvar.grid(column=0, row=4)  # Размещаем виджет
            except Exception as e:  # Обрабатываем ошибки при загрузке изображения
                print(f"[ERROR] Unable to load image: {e}")

    def OpenImage(self, filepath):  # Метод для открытия изображения (не используется)
        print(f"[INFO] Opening image: {filepath}")  # Выводим путь изображения

# Главная точка входа в приложение
if __name__ == '__main__':
    root = Root()  # Создаем экземпляр основного окна
    root.mainloop()  # Запускаем цикл обработки событий
