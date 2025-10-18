import cv2
import numpy as np
import os
# Проверяем, есть ли YOLOv8 (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Библиотека ultralytics не найдена. YOLOv8 недоступен. Установите командой: pip install ultralytics")
# === YOLOv4 реализация ===
def apply_yolo_object_detection(image_to_process):
    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for _ in range(3))
    object_count = 0
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                x = int(center_x - obj_width / 2)
                y = int(center_y - obj_height / 2)
                boxes.append([x, y, obj_width, obj_height])
                class_indexes.append(class_id)
                class_scores.append(float(confidence))
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]
        if classes[class_index] in classes_to_look_for:
            object_count += 1
            image_to_process = draw_object_bounding_box(image_to_process, class_index, box)
    final_image = draw_object_count(image_to_process, object_count)
    return final_image
def draw_object_bounding_box(image_to_process, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)
    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)
    return final_image
def draw_object_count(image_to_process, object_count):
    start = (45, 150)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = f"Objects found: {object_count}"
    color = (255, 255, 255)
    final_image = cv2.putText(image_to_process, text, start, font, font_size, color, width, cv2.LINE_AA)
    return final_image
# === Запуск на изображении (YOLOv4 или YOLOv8) ===
def start_image_object_detection(use_yolov8=False):
    pick = input("Введите путь к изображению: ").strip()
    image = cv2.imread(pick)
    if image is None:
        print("Ошибка: не удалось открыть изображение.")
        return
    if use_yolov8:
        model = YOLO("yolov8n.pt")
        results = model(pick)
        # Получаем изображение с нарисованными боксами
        annotated_frame = results[0].plot()
        # Показываем и держим окно открытым, пока не нажмешь клавишу
        cv2.imshow("YOLOv8 Image Detection", annotated_frame)
        print("Нажми любую клавишу, чтобы закрыть окно.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = apply_yolo_object_detection(image)
        cv2.imshow("YOLOv4-tiny Image Detection", image)
        print("Нажми любую клавишу, чтобы закрыть окно.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# === Запуск видео ===
def start_video_object_detection(use_yolov8=False):
    pick = input("Введите путь к видеофайлу или 0 для веб-камеры: ").strip()
    if pick == "0":
        pick = 0
    video_capture = cv2.VideoCapture(pick)
    if not video_capture.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return
    print("Запущено видео-распознавание. Нажмите 'q' для выхода.")
    if use_yolov8:
        model = YOLO("yolov8n.pt")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            results = model(frame, stream=True)
            for r in results:
                annotated_frame = r.plot()
                cv2.imshow("YOLOv8 Video Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = apply_yolo_object_detection(frame)
            cv2.imshow("YOLOv4-tiny Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.release()
    cv2.destroyAllWindows()
# === Основной блок ===
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_path, "yolov4-tiny.cfg")
    weights_path = os.path.join(base_path, "yolov4-tiny.weights")
    # Загружаем YOLOv4
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    out_layers_index = net.getUnconnectedOutLayers()
    out_layers = [layer_names[i - 1] for i in out_layers_index]
    with open("coco.names.txt") as file:
        classes = file.read().split("\n")
    classes_to_look_for = classes  # можно ограничить список
    print("\nВыберите версию модели:")
    print("1 - YOLOv4-tiny (твоя версия)")
    if YOLO_AVAILABLE:
        print("2 - YOLOv8 (новая, лучше и проще)")
    choice_model = input("Введите 1 или 2: ").strip()
    use_yolov8 = choice_model == "2" and YOLO_AVAILABLE
    print("\nВыберите режим:")
    print("1 - Распознавание на изображении")
    print("2 - Распознавание на видео")
    choice = input("Введите 1 или 2: ").strip()
    if choice == "2":
        start_video_object_detection(use_yolov8)
    else:
        start_image_object_detection(use_yolov8)
