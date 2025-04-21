import face_recognition
import cv2, os
import numpy as np
import time

# --- Конфигурация для оптимизации ---
PROCESS_EVERY_N_FRAMES = 2 # Обрабатывать каждый N-й кадр
RESIZE_FRAME_FACTOR = 0.5 # Фактор уменьшения размера кадра (1.0 для оригинального размера)
# --- Конец конфигурации ---

IMGS_DIR = os.path.join('Face_recognition_lib', 'src', 'img')
ENCODINGS_FILE = os.path.join('Face_recognition_lib', 'src', 'known_faces.npy')
NAMES_FILE = os.path.join('Face_recognition_lib', 'src', 'known_names.npy')

known_face_encodings = []
known_face_names = []

# --- Оптимизация: Загрузка или создание кодировок известных лиц ---
if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
    print(f"Загрузка известных лиц из файлов: {ENCODINGS_FILE}, {NAMES_FILE}")
    try:
        known_face_encodings = np.load(ENCODINGS_FILE, allow_pickle=True)
        known_face_names = np.load(NAMES_FILE, allow_pickle=True)
        print(f"Успешно загружено {len(known_face_encodings)} известных лиц из файлов.")
    except Exception as e:
        print(f"Ошибка при загрузке известных лиц из файлов: {e}. Повторная загрузка из изображений.")
        known_face_encodings = []
        known_face_names = []

if not known_face_encodings:
    print(f"Загрузка известных лиц из директории: {IMGS_DIR}")

    if not os.path.exists(IMGS_DIR):
        print(f"Ошибка: Директория с известными лицами не найдена: '{IMGS_DIR}'")
        print("Пожалуйста, создайте эту директорию и поместите туда изображения.")
        exit()

    for filename in os.listdir(IMGS_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMGS_DIR, filename)
            name = os.path.splitext(filename)[0]

            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Загружено лицо: {name}")
                else:
                    print(f"Лицо не найдено в файле: {filename}")

            except Exception as e:
                print(f"Ошибка обработки файла {filename}: {e}")

    if not known_face_encodings:
        print("Ошибка: Не найдено известных лиц для сравнения. Проверьте директорию и изображения.")
        exit()
    else:
        # Сохраняем кодировки и имена для более быстрой загрузки в будущем
        try:
            os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
            np.save(ENCODINGS_FILE, known_face_encodings)
            np.save(NAMES_FILE, known_face_names)
            print(f"Кодировки и имена известных лиц сохранены в {os.path.dirname(ENCODINGS_FILE)}")
        except Exception as e:
            print(f"Предупреждение: Не удалось сохранить кодировки и имена известных лиц: {e}")


print(f"Всего загружено {len(known_face_encodings)} известных лиц.")

video_capture = cv2.VideoCapture(0) # Используем веб-камеру по умолчанию (индекс 0)

if not video_capture.isOpened():
    print("Ошибка: Не удалось получить доступ к веб-камере.")
    exit()

print("Видеопоток запущен. Нажмите 'q' для выхода.")

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Не удалось прочитать кадр с веб-камеры.")
        break # Выходим из цикла, если кадр не считан

    frame_count += 1

    # --- Оптимизация: Пропускаем кадры ---
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        cv2.imshow('Video', frame) # Показываем текущий кадр без обработки
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- Оптимизация: Изменение размера кадра ---
    if RESIZE_FRAME_FACTOR != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FRAME_FACTOR, fy=RESIZE_FRAME_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Находим все лица и их кодировки в текущем (возможно, уменьшенном) кадре
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Перебираем каждое найденное лицо в кадре
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Увеличиваем координаты обратно, если кадр был уменьшен
        if RESIZE_FRAME_FACTOR != 1.0:
            top = int(top / RESIZE_FRAME_FACTOR)
            right = int(right / RESIZE_FRAME_FACTOR)
            bottom = int(bottom / RESIZE_FRAME_FACTOR)
            left = int(left / RESIZE_FRAME_FACTOR)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Неизвестный"

        # --- Оптимизация: Использование numpy.argmin для нахождения лучшего совпадения ---
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
             # Можно добавить порог для face_distance, чтобы отфильтровать плохие совпадения
             # if face_distances[best_match_index] < tolerance: # tolerance - настройте пороговое значение
            name = known_face_names[best_match_index]


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Красный прямоугольник

        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1) # Белый текст

    # Показываем результирующий кадр
    cv2.imshow('Video', frame)

    # Выходим из цикла, если нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

print("Программа завершена.")

# --- Дополнительные идеи для оптимизации ---
# 1. Использование более быстрого детектора лиц: Вместо face_recognition.face_locations
#    можно использовать каскады Хаара OpenCV или модели на основе глубокого обучения
#    (например, SSD) для обнаружения лиц, а затем передавать найденные области в
#    face_recognition.face_encodings.
# 2. Использование GPU: Если доступен CUDA-совместимый GPU, можно использовать
#    версию dlib с поддержкой CUDA или другие библиотеки для ускорения вычислений
#    кодировок лиц.
# 3. Оптимизация сравнения лиц: Для большого количества известных лиц можно использовать
#    структуры данных для поиска ближайших соседей, такие как KD-деревья или Ball Tree,
#    для более быстрого поиска совпадений вместо линейного сравнения с каждым известным лицом.
# 4. Многопоточность/Асинхронность: Обработка видеопотока в одном потоке и выполнение
#    распознавания в другом может улучшить отзывчивость интерфейса, но требует более
#    сложной синхронизации.
# 5. Уменьшение MAX_LEN для кодирования лиц, если это возможно без существенной потери точности.
#    (Хотя в данном коде нет явного MAX_LEN для кодирования лиц, это относится к общим подходам).