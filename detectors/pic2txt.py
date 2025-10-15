#!/usr/bin/env python3

from PIL import Image
import pytesseract
import numpy as np
import argparse
import sys
import cv2


# Вспомогательные функции I/O

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return img


# Геометрия и бинаризация

def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

def binarize_document(gray: np.ndarray) -> np.ndarray:
    """
    Универсальная бинаризация документов с легкой чисткой
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = adjust_contrast_brightness(gray, contrast=2.0)
    g = clahe.apply(g)
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                          cv2.THRESH_BINARY, 11, 9)

    # Уборка мелких шумов
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((1, 2), np.uint8))
    return g

def remove_lines_and_decor(th: np.ndarray) -> np.ndarray:
    if th.dtype != np.uint8:
        th = th.astype(np.uint8)

    inv = 255 - th
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vert  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))
    lines = cv2.bitwise_or(horiz, vert)

    # Тонкая маска для сохранения буковок
    lines = cv2.erode(lines, np.ones((2, 2), np.uint8), iterations=1)

    # Финальная уборка линий
    cleaned = cv2.bitwise_and(th, 255 - lines)
    return cleaned


def binarize_newspaper(gray: np.ndarray) -> np.ndarray:
    """
    Режим для газет, агрессивный контраст, удаляем декоративные эл-ты
    """
    th = binarize_document(gray)
    th = remove_lines_and_decor(th)
    return th


def preprocess(img_bgr: np.ndarray, mode: str, scale: float = 1.5) -> np.ndarray:
    """
    Предобработка под распознавание
    """
    if scale and scale != 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # режимы
    if mode == "document":
        th = binarize_document(gray)
    elif mode == "newspaper":
        th = binarize_newspaper(gray)
    elif mode == "inverted":
        th = binarize_document(255 - gray)
    elif mode == "comic":
        th = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   25, 15)
        th = remove_lines_and_decor(th)
    else:  # default
        if np.mean(gray) < 110:
            th = binarize_document(255 - gray)
        else:
            th = binarize_document(gray)

    return th


def perform_ocr(img_bgr: np.ndarray, mode: str, lang: str, psm: int = 6, scale: float = 1.5) -> str:
    th = preprocess(img_bgr, mode=mode, scale=scale)
    pil_img = Image.fromarray(th)

    config = f"--psm {psm} -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
    return text.strip()


# Клиентская часть, парсинг и всякая шляпа

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Чтение текста с картинки")
    p.add_argument("image", help="Путь к изображению или URL")
    p.add_argument("--mode",
                  choices=["default", "document", "newspaper", "inverted", "comic"],
                  default="default",
                  help="Режимы предобработки")
    p.add_argument("--lang", default="eng", help="Языки для Tesseract, например: eng+rus")
    p.add_argument("--tesseract-cmd", help="Путь к исполняемому файлу tesseract")
    p.add_argument("--psm", type=int, default=1, help="Page Segmentation Mode для Tesseract (по умолчанию 1)")
    p.add_argument("--show", action="store_true", help="Показать предобработанное изображение")
    p.add_argument("--output", "-o", help="Сохранить результат в файл")
    p.add_argument("--scale", "-s", type=float, default=1.5, help="Коэффицент изменения размера")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        img = load_image(args.image)
    except Exception as e:
        print(f"Ошибка загрузки: {e}", file=sys.stderr)
        return 1

    try:
        text = perform_ocr(img, mode=args.mode, lang=args.lang, psm=args.psm, 
                           scale = args.scale)
    except Exception as e:
        print(f"Ошибка OCR: {e}", file=sys.stderr)
        return 1

    if args.show:
        try:
            preview = preprocess(img, mode=args.mode, scale=args.scale)
            cv2.imshow("preprocessed", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Не удалось показать изображение: {e}", file=sys.stderr)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Результат сохранён в {args.output}")
        except Exception as e:
            print(f"Не удалось сохранить файл: {e}", file=sys.stderr)
            return 1
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
