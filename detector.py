#!/usr/bin/env python3

import json
import os
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
import langdetect

# Импорт функций EAST для детекции боксов
from detectors.text_detection import (
    get_new_size,
    decode_predictions,
    merge_boxes
)

# Импорт функций для обработки изображений и OCR
from detectors.pic2txt import (
    load_image,
    preprocess,
    perform_ocr
)

def detect_language(text: str) -> str:
    """
    Определяет язык текста на основе анализа.
    Возвращает 'unknown', если текст слишком короткий или не определён.
    """
    if not text or len(text.strip()) < 3:
        return "unknown"

    detected = langdetect.detect(text)
    return {'ru': 'russian', 'en': 'english'}.get(detected, "unknown")

def extract_text_from_regions(image, boxes, image_path: str, output_dir: str) -> list:
    """
    Извлекает текст из детектированных EAST регионов с использованием Tesseract.
    Применяет предобработку и пробует разные режимы через perform_ocr.
    """
    texts_by_region = []
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    H, W = image.shape[:2]

    for i, (startX, startY, endX, endY) in enumerate(boxes):
        # Извлечение области изображения
        roi = image[max(0, startY):min(H, endY), max(0, startX):min(W, endX)]
        if roi.size == 0:
            continue

        # Сохранение исходного региона для отладки
        crop_path = os.path.join(crops_dir, f"region_{i}.jpg")
        cv2.imwrite(crop_path, roi)

        # Применение OCR с предобработкой
        region_text = ""
        modes = ["newspaper", "document"]
        for mode in modes:
            try:
                region_text = perform_ocr(roi, mode=mode, lang="eng+rus", scale=2.0)
                if region_text.strip():
                    break
            except Exception:
                continue

        # Сохранение результата, если текст найден
        if region_text.strip():
            region_info = {
                "region_id": i,
                "bbox": [int(startX), int(startY), int(endX), int(endY)],
                "crop_path": crop_path,
                "text": region_text.strip(),
                "language": detect_language(region_text)
            }
            texts_by_region.append(region_info)
            print(f"Регион {i}: '{region_text.strip()[:50]}...'")
        else:
            print(f"егион {i}: текст не распознан")
    return texts_by_region

def backup_full_image_ocr(image_path: str, output_dir: str) -> dict:
    """
    Резервный OCR для всего изображения, если регионы не найдены.
    """
    img = load_image(image_path)
    for mode in ["newspaper", "document"]:
        try:
            text = perform_ocr(img, mode=mode, lang="eng+rus", scale=2.0)
            if text.strip():
                return {
                    "region_id": -1,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "crop_path": "",
                    "text": text.strip(),
                    "language": detect_language(text)
                }
        except:
            continue
    return None

def process_image(image_path: str, output_dir: str = "results") -> dict:
    """
    Главная функция: использует EAST для детекции боксов и Tesseract для текста.
    Сохраняет отладочные данные и JSON с результатами.
    """
    try:
        image = load_image(image_path)
        orig = image.copy()
        (H, W) = image.shape[:2]
        print(f"   Размер: {W}x{H}")

        # Подготовка изображения для EAST
        (W_new, H_new) = get_new_size(W, H)
        print(f"   EAST размер: {W_new}x{H_new}")
        image_resized = cv2.resize(image, (W_new, H_new))

        # Загрузка и выполнение EAST
        EAST_MODEL_PATH = "detectors/models/frozen_east_text_detection.pb"
        if not os.path.exists(EAST_MODEL_PATH):
            raise FileNotFoundError(f"Модель EAST: {EAST_MODEL_PATH}")
        net = cv2.dnn.readNet(EAST_MODEL_PATH)
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (W_new, H_new),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        mapOutputs = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        boxes, confidences = decode_predictions(*mapOutputs, W_new, H_new,
                                               min_confidence=0.05, width_threshold=0.005)
        print(f"Найдено боксов: {len(boxes)}")

        # NMS и merge для устранения дубликатов
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.01, 0.95)
        if len(indices) == 0:
            indices = np.array([i for i, conf in enumerate(confidences) if conf > 0.1])
        selected_boxes = [boxes[i] for i in indices.flatten()]
        merged_boxes, _ = merge_boxes(selected_boxes, confidences, max_dist=150)
        rW, rH = W / float(W_new), H / float(H_new)
        merged_boxes_scaled = [(int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH))
                               for startX, startY, endX, endY in merged_boxes]

        # Сохранение отладочного изображения с боксами
        cv2.imwrite(os.path.join(output_dir, "debug_boxes.jpg"), orig)

        # Инициализация результата
        result = {
            "input_image": image_path,
            "image_size": {"width": W, "height": H},
            "text_regions": [],
            "total_regions": 0,
            "processing_date": "2025-10-15",
            "east_debug": {"raw_boxes": len(boxes)}
        }

        # Обработка регионов
        if merged_boxes_scaled:
            result["text_regions"] = extract_text_from_regions(orig, merged_boxes_scaled, image_path, output_dir)
            result["total_regions"] = len(result["text_regions"])
        if result["total_regions"] == 0:
            backup = backup_full_image_ocr(image_path, output_dir)
            if backup:
                result["text_regions"], result["total_regions"] = [backup], 1

        # Объединение текста и определение языка
        if result["total_regions"] > 0:
            result["full_text"] = " ".join([r["text"] for r in result["text_regions"]])
            result["detected_language"] = detect_language(result["full_text"])

        # Сохранение результатов
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "text_detection.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    except Exception as e:
        print(f"Ошибка: {e}")
        return {"error": str(e), "input_image": image_path, "status": "failed"}

def main():
    """
    Точка входа: принимает путь к изображению и опциональный output_dir.
    """
    if len(sys.argv) < 2:
        sys.exit(1)
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        sys.exit(1)

    result = process_image(image_path, output_dir)

    print(f"   Регионов текста: {result.get('total_regions', 0)}")
    print(f"   Язык: {result.get('detected_language', 'unknown')}")
    print(f"   JSON: {os.path.join(output_dir, 'text_detection.json')}")
    if result.get('total_regions', 0) > 0:
        print(f"   Текст: {result.get('full_text', '')[:100]}{'...' if len(result.get('full_text', '')) > 100 else ''}")

if __name__ == "__main__":
    main()