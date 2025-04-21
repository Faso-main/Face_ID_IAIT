import face_recognition
import cv2, os
import numpy as np


GALKIN1_IMG=os.path.join('Face_recognition_lib','src','img','Tavkevic.jpg')

try:
    known_image = face_recognition.load_image_file(GALKIN1_IMG)
    known_encoding = face_recognition.face_encodings(known_image)[0]
except IndexError:
    print("Не удалось найти лицо на изображении")
    exit()

# Создаем массивы для известных лиц
known_face_encodings = [known_encoding]
known_face_names = ["Tavkevic"]

# Инициализируем видеопоток
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Конвертируем BGR в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Находим все лица
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Получаем кодировки только если найдены лица
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Неизвестный"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()