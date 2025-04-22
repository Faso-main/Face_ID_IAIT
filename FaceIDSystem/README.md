# Сравнение алгоритмов детекции лицевых ориентиров

## 1. Active Shape Models (ASM) in dlib
**Описание**:  
Статистическая модель формы, использующая метод главных компонент (PCA) для представления допустимых вариаций формы лица. Итеративно подстраивается к границам объектов.

**Преимущества**:
- Высокая скорость обработки (120+ FPS)
- Хорошая точность для фронтальных лиц
- Минимальные зависимости (требуется только dlib)

**Недостатки**:
- Чувствительность к начальной инициализации
- Плохая работа с профильными ракурсами
- Ограниченное количество точек (68)

**Модель**:  
[dlib 68-face-landmarks model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)  
**Документация**:  
[Official dlib ASM documentation](http://dlib.net/face_landmark_detection_ex.cpp.html)

**Примеры github**:  
- https://github.com/johnwmillr/ActiveShapeModels
- https://github.com/YashGunjal/asm

---

## 2. Discriminative Response Map Fitting (DRMF)
**Описание**:  
Метод на основе ансамблей регрессионных деревьев, предсказывающий позиции точек по откликам фильтров Хаара.

**Преимущества**:
- Лучшая точность для нестандартных ракурсов
- Поддержка большего количества точек (76+)
- Устойчивость к частичным окклюзиям

**Недостатки**:
- Требует больше вычислительных ресурсов (45-60 FPS)
- Сложность в настройке параметров
- Менее стабильная работа при слабом освещении

**Модель**:  
[DRMF model for dlib](https://github.com/tzutalin/dlib-android/)  
**Исследование**:  
[DRMF Original Paper](https://ieeexplore.ieee.org/document/6130296)

---

## 3. Constrained Local Models (CLM) in OpenCV
**Описание**:  
Комбинация глобальной модели формы с локальными экспертами для каждого ориентира. Использует ограниченную оптимизацию.

**Преимущества**:
- Лучшая точность для выражений лица
- Устойчивость к изменениям мимики
- Поддержка динамической коррекции точек

**Недостатки**:
- Низкая производительность (30-40 FPS)
- Требует точной инициализации
- Сложная интеграция (требует opencv-contrib)

**Модель**:  
[OpenCV LBF model](https://github.com/kurnianggoro/GSOC2017)  
**Документация**:  
[OpenCV Facemark API](https://docs.opencv.org/4.x)
**Ссылка на модель**
[lbfmodel](https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml)

---

## Рекомендации по выбору

1. **Для реального времени** → ASM (dlib)
2. **Для профильных лиц** → DRMF 
3. **Для анализа эмоций** → CLM
4. **Максимальная точность** → Комбинация ASM+CLM

**Универсальное решение**:
```python
import dlib
import cv2

# Инициализация всех моделей
asm_predictor = dlib.shape_predictor("shape_predictor_68.dat")
drmf_predictor = dlib.shape_predictor("drmf_model.dat") 
clm = cv2.face.createFacemarkLBF().loadModel("lbfmodel.yaml")