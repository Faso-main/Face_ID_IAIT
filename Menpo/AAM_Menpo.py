import cv2
import os
import menpo.io as mio
from menpo.shape import bounding_box
import menpofit.aam
import numpy as np


IMG_PATH = os.path.join('src','AAM', 'img', 'Galkin2.jpg')

# 1 Конвертация в numpy array
im = mio.import_image(IMG_PATH)
im_gray = im.as_greyscale()

# 2 Конвертация в openCV
img_np = im_gray.pixels_with_channels_at_back().squeeze().astype(np.uint8)

# 3 Детекция 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0: print("Лицо не обнаружено")
else:
    x, y, w, h = faces[0]
    bb = bounding_box((x, y), (x + w, y + h))
    
    # 4 Загрузка AAM-фиттера
    aam_fitter = menpofit.aam.load_balanced_frontal_face_fitter()
    
    # 5 Подгонка и визуализация
    result = aam_fitter.fit_from_bb(im_gray, bb, max_iters=50)
    result.view()