import os
import cv2
import numpy as np


def get_new_size(original_width, original_height):
    """
    Определяет новый размер изображения для обработки моделью EAST.
    
    Parameters
    ----------
    original_width : int
        Ширина исходного изображения в пикселях
    original_height : int
        Высота исходного изображения в пикселях
        
    Returns
    -------
    tuple
        Кортеж (new_width, new_height) с новыми размерами, кратными 32
        
    Raises
    ------
    ValueError
        Если не удалось рассчитать размер изображения
    """
    try:
        base_width = 1280
        ratio = base_width / float(original_width)
        new_height = int(original_height * ratio)
        
        new_width = (base_width // 32) * 32
        new_height = (new_height // 32) * 32
        
        new_width = max(320, new_width)
        new_height = max(320, new_height)
        
        print(f"Размер оригинала: {original_width}x{original_height}")
        print(f"Размер для обработки: {new_width}x{new_height}")
        
        return (new_width, new_height)
    except Exception as e:
        raise ValueError(f"Ошибка расчета размера: {str(e)}")


def decode_predictions(scores, geometry, image_width, image_height, min_confidence=0.2, width_threshold=0.2):
    """
    Декодирует выходные данные модели EAST в bounding boxes.
    
    Parameters
    ----------
    scores : numpy.ndarray
        Выход модели EAST с вероятностями наличия текста
    geometry : numpy.ndarray
        Выход модели EAST с геометрическими параметрами текста
    image_width : int
        Ширина изображения для обработки
    image_height : int
        Высота изображения для обработки
    min_confidence : float, optional
        Минимальная уверенность для детекции текста (по умолчанию 0.2)
    width_threshold : float, optional
        Минимальная ширина текстовой области относительно ширины изображения
        
    Returns
    -------
    tuple
        Кортеж (boxes, confidences) со списками bounding boxes и уверенностей
        
    Raises
    ------
    ValueError
        Если не удалось декодировать предсказания модели
    """
    try:
        (numRows, numCols) = scores.shape[2:4]
        boxes = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            x0, x1, x2, x3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
            angles = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < min_confidence:
                    continue
                
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = angles[x]
                cosA = np.cos(angle)
                sinA = np.sin(angle)
                
                h = x0[x] + x2[x]
                w = x1[x] + x3[x]
                
                if w < width_threshold * float(image_width):
                    continue
                    
                endX = int(offsetX + (cosA * x1[x]) + (sinA * x2[x]))
                endY = int(offsetY - (sinA * x1[x]) + (cosA * x2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                padding_w = int(w * 0.5)
                padding_h = int(h * 0.5)
                startX = max(0, startX - padding_w)
                startY = max(0, startY - padding_h)
                endX = min(image_width, endX + padding_w)
                endY = min(image_height, endY + padding_h)
                
                boxes.append((startX, startY, endX, endY))
                confidences.append(float(scoresData[x]))

        return (boxes, confidences)
    except Exception as e:
        raise ValueError(f"Ошибка декодирования предсказаний: {str(e)}")


def merge_boxes(boxes, confidences, overlap_threshold=0.3, max_dist=50):
    """
    Объединяет близко расположенные bounding boxes в более крупные области.
    
    Parameters
    ----------
    boxes : list
        Список bounding boxes в формате (startX, startY, endX, endY)
    confidences : list
        Список уверенностей для каждого bounding box
    overlap_threshold : float, optional
        Порог перекрытия для объединения боксов (по умолчанию 0.3)
    max_dist : int, optional
        Максимальное расстояние между боксами для объединения (по умолчанию 50)
        
    Returns
    -------
    tuple
        Кортеж (merged_boxes, merged_conf) с объединенными боксами и уверенностями
        
    Raises
    ------
    ValueError
        Если не удалось объединить bounding boxes
    """
    try:
        if len(boxes) == 0:
            return [], []

        indices = np.argsort([box[1] for box in boxes])
        sorted_boxes = [boxes[i] for i in indices]
        sorted_conf = [confidences[i] for i in indices]

        merged_boxes = []
        merged_conf = []
        current_box = sorted_boxes[0]
        current_conf = sorted_conf[0]

        for i in range(1, len(sorted_boxes)):
            next_box = sorted_boxes[i]
            next_conf = sorted_conf[i]

            if abs(current_box[1] - next_box[1]) < max_dist and abs(current_box[3] - next_box[3]) < max_dist:
                if current_box[2] >= next_box[0] - max_dist:
                    current_box = (min(current_box[0], next_box[0]),
                                   min(current_box[1], next_box[1]),
                                   max(current_box[2], next_box[2]),
                                   max(current_box[3], next_box[3]))
                    current_conf = max(current_conf, next_conf)
                    continue

            merged_boxes.append(current_box)
            merged_conf.append(current_conf)
            current_box = next_box
            current_conf = next_conf

        merged_boxes.append(current_box)
        merged_conf.append(current_conf)

        return merged_boxes, merged_conf
    except Exception as e:
        raise ValueError(f"Ошибка объединения боксов: {str(e)}")


def draw_boxes_and_save(image, boxes, confidences, scale_factors, output_path="results/out_image.jpg"):
    """
    Визуализирует обнаруженные текстовые области на изображении.
    
    Parameters
    ----------
    image : numpy.ndarray
        Исходное изображение в формате OpenCV
    boxes : list
        Список bounding boxes для отрисовки
    confidences : list
        Список уверенностей для подписей на bounding boxes
    scale_factors : tuple
        Коэффициенты масштабирования (rW, rH) для пересчета координат
    output_path : str, optional
        Путь для сохранения результата (по умолчанию "results/out_image.jpg")
        
    Raises
    ------
    ValueError
        Если не удалось отрисовать и сохранить результат
    """
    try:
        W, H = image.shape[1], image.shape[0]
        rW, rH = scale_factors
        
        for i, box in enumerate(boxes):
            startX, startY, endX, endY = box
            
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            
            startX = max(0, min(startX, W))
            startY = max(0, min(startY, H))
            endX = max(startX, min(endX, W))
            endY = max(startY, min(endY, H))
            
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            confidence = confidences[i]
            label = f"{confidence:.2f}"
            
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (startX, startY - label_height - baseline - 5), 
                          (startX + label_width, startY), (0, 255, 0), -1)
            cv2.putText(image, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, image)
        print(f"Результат сохранен в {output_path}")

        cv2.imshow("EAST Text Detection - Merged", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        raise ValueError(f"Ошибка отрисовки и сохранения: {str(e)}")


def main():
    """
    Основная функция для обработки изображения и обнаружения текста.
    
    Workflow
    --------
    1. Чтение входных параметров (путь к изображению и выходному файлу)
    2. Загрузка и предобработка изображения
    3. Загрузка модели EAST для детекции текста
    4. Выполнение предсказания и декодирование результатов
    5. Объединение близких bounding boxes
    6. Визуализация и сохранение результата
    
    Raises
    ------
    FileNotFoundError
        Если модель EAST или входное изображение не найдены
    ValueError
        Если произошла ошибка в процессе обработки изображения
    """
    try:
        input_data = input().split()
        
        if len(input_data) < 1:
            print("Ошибка: Не указан путь к входному изображению")
            return
        
        image_path = input_data[0]
        
        if len(input_data) > 1:
            file_out_image = input_data[1]
        else:
            file_out_image = "results/out_image.jpg"
        
        output_dir = os.path.dirname(file_out_image)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Создана директория: {output_dir}")
        
        EAST_MODEL_PATH = "models/frozen_east_text_detection.pb"
        
        if not os.path.exists(EAST_MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена: {EAST_MODEL_PATH}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        orig = image.copy()
        (H, W) = image.shape[:2]

        (W_new, H_new) = get_new_size(W, H)
        image_resized = cv2.resize(image, (W_new, H_new))

        net = cv2.dnn.readNet(EAST_MODEL_PATH)

        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (W_new, H_new), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        mapOutputs = net.forward(layerNames)

        (scores, geometry) = mapOutputs
        boxes, confidences = decode_predictions(scores, geometry, W_new, H_new, 
                                               min_confidence=0.2, width_threshold=0.02)

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.9)
            
            if len(indices) == 0:
                high_conf_indices = [i for i, conf in enumerate(confidences) if conf > 0.3]
                indices = np.array(high_conf_indices) if high_conf_indices else []
            
            selected_boxes = [boxes[i] for i in indices.flatten()]
            selected_conf = [confidences[i] for i in indices.flatten()]
            
            merged_boxes, merged_conf = merge_boxes(selected_boxes, selected_conf, max_dist=100)
            
            scale_factors = (W / float(W_new), H / float(H_new))
            draw_boxes_and_save(orig, merged_boxes, merged_conf, scale_factors, file_out_image)
        else:
            print("Текст не обнаружен на изображении")
            
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        return


if __name__ == "__main__":
    main()