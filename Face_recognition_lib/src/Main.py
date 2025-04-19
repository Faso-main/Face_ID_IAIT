import face_recognition
import cv2, os
import numpy as np


REG_IMG=os.path.join('Face_recognition_lib','src','img','reg_img.jpg')

# Загружаем изображения для распознавания 
known_image = face_recognition.load_image_file(REG_IMG)
known_encoding = face_recognition.face_encodings(known_image)[0]

# Создаем массивы для известных лиц и их имен
known_face_encodings = [known_encoding]
known_face_names = ["Faso"]

# Инициализируем видеопоток 
video_capture = cv2.VideoCapture(0)

while True:
    # Захватываем кадр из видеопотока
    ret, frame = video_capture.read()
    
    # Конвертируем изображение из BGR (который использует OpenCV) в RGB (который использует face_recognition)
    rgb_frame = frame[:, :, ::-1]
    
    # Находим все лица и их кодировки в текущем кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Проходим по всем найденным лицам в кадре
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Сравниваем лицо с известными лицами
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Используем лицо с наименьшим расстоянием (наиболее похожее)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Рисуем прямоугольник вокруг лица и подписываем его
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Отображаем результат
    cv2.imshow('Рабочее окно', frame)
    
    # Выход из цикла по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()