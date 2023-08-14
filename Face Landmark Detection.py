import cv2
import mediapipe as mp

# Инициализация объекта FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
# Инициализация объекта MediaPipe для ввода видеопотока
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Захват видеопотока с веб-камеры
cap = cv2.VideoCapture(0)
while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Перевод изображения из BGR в RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Обнаружение лиц и определение landmark'ов
    results = face_mesh.process(image_rgb)

    # Отображение landmark'ов на изображении
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Отображение изображения
    cv2.imshow('Face Landmarks Detection', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()