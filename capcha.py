import cv2
import numpy as np
import os
def apply_yolo_object_detection(image_to_process):
    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608,608), (0,0,0),swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range (3))
    object_count = 0
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # ← фильтруем только уверенные детекции
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                x = int(center_x - obj_width / 2)
                y = int(center_y - obj_height / 2)
                boxes.append([x, y, obj_width, obj_height])
                class_indexes.append(class_id)
                class_scores.append(float(confidence))
    chosen_boxes =cv2.dnn.NMSBoxes( boxes, class_scores,0.0,0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]
        box = boxes[box_index]
        class_index= class_indexes[box_index]
        if classes[class_index] in classes_to_look_for:
            object_count += 1
            image_to_process = draw_object_bounding_box(image_to_process, class_index,box)
    final_image = draw_object_count(image_to_process, object_count)
    return final_image
def draw_object_bounding_box(image_to_process, index ,box):
    x,y,w,h = box
    start = (x,y)
    end = (x+w,y+h)
    color = (0,255,0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width) 
    start = (x,y-10)
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
    text = "" 
    white_color =(255,255,255)
    black_outline_color = (0,0,0)
    final_image = cv2.putText(image_to_process, text, start,font,font_size, black_outline_color,width, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start,font,font_size, white_color,width, cv2.LINE_AA)
    return final_image
def start_image_object_detection():
    pick = input("Введите путь к изображению: ").strip()
    try:
        image = cv2.imread(pick)
        if image is None:
            print("Ошибка: не удалось открыть изображение.")
            exit()
        image = apply_yolo_object_detection(image)
        cv2.imshow("Image Object Detection", image)
        if cv2.waitKey(0) :
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
def start_video_object_detection():
    pick = input("Введите путь к видеофайлу или 0 для веб-камеры: ").strip()
    if pick == "0":
        pick = 0  # Используем веб-камеру
    video_capture = cv2.VideoCapture(pick)
    if not video_capture.isOpened():
        print("Ошибка: не удалось открыть источник видео.")
        return
    print("Запущено видео-распознавание объектов. Нажмите 'q' для выхода.")
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("⚠️ Видео закончилось или источник недоступен.")
                break
            frame = apply_yolo_object_detection(frame)
            cv2.imshow("Video Object Detection", frame)
            # Клавиша 'q' — выход
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("завершение работы по нажатию 'q'.")
                break
    except KeyboardInterrupt:
        print("Прервано пользователем.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_path, "yolov4-tiny.cfg")
    weights_path = os.path.join(base_path, "yolov4-tiny.weights")
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    out_layers_index = net.getUnconnectedOutLayers()
    out_layers = [layer_names[i - 1] for i in out_layers_index]
    with open("coco.names.txt") as file:
        classes = file.read().split("\n")
    classes_to_look_for = ["person", "bicycle", "car", "motorbike", "aeroplane", 
                           "bus", "train", "truck", "boat", "traffic light", 
                           "fire hydrant", "stop sign", "parking meter", "bench", 
                           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
                           "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", 
                           "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                           "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                           "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                           "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
                           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",]
    print("choose image or video object detection:")
    print("1 - image object detection")
    print("2 - video object detection")
    choice = input("Enter 1 or 2: ")
    if choice == "2":
        start_video_object_detection()
    else:
        start_image_object_detection()