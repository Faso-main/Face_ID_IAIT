import face_recognition
import cv2, os
import numpy as np


IMGS_DIR = os.path.join('Face_recognition_lib', 'src', 'img')

# Проверяем, существует ли директория
if not os.path.exists(IMGS_DIR):
    print(f"Ошибка: Директория с известными лицами не найдена: '{IMGS_DIR}'")
    print("Пожалуйста, создайте эту директорию и поместите туда изображения.")
    exit()

known_face_encodings = []
known_face_names = []

print(f"Загрузка известных лиц из директории: {IMGS_DIR}")

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

print(f"Успешно загружено {len(known_face_encodings)} известных лиц.")

video_capture = cv2.VideoCapture(0) # Используем веб-камеру по умолчанию (индекс 0)

if not video_capture.isOpened():
    print("Ошибка: Не удалось получить доступ к веб-камере.")
    exit()

print("Видеопоток запущен. Нажмите 'q' для выхода.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Не удалось прочитать кадр с веб-камеры.")
        break # Выходим из цикла, если кадр не считан

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Находим все лица и их кодировки в текущем кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Перебираем каждое найденное лицо в кадре
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Неизвестный" # Имя по умолчанию, если совпадение не найдено

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        # Если лучшее совпадение действительно считается совпадением (согласно matches)
        if matches[best_match_index]:
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