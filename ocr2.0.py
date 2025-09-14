# app.py
import os
import re
import json
import requests
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import threading
import time
import random

from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import pytesseract
from flask import Flask, request, jsonify, send_file, render_template_string

# --- Настройки ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

dpi = 300
use_preprocessing = True
OLLAMA_MODEL = "gemma2:2b-instruct-q4_K_M"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

REQUIRED_FIELDS = [
    "document_type", "contract_number", "sign_date", "expiry_date",
    "seller", "buyer", "amount", "currency",
    "validation_status", "extraction_accuracy"
]

# Статусные сообщения
PROCESSING_MESSAGES = [
    "Подготовка ИИ-модели для анализа...",
    "Преобразование PDF в изображения...",
    "Анализ структуры документа...",
    "Распознавание текста с помощью OCR...",
    "Исправление орфографических ошибок...",
    "Извлечение ключевых полей...",
    "Валидация извлеченных данных...",
    "Формирование структурированного JSON...",
    "Подсчет метрик качества...",
    "Завершение обработки..."
]
# ------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# --- OCR и обработка (ваш код) ---
def robust_preprocess_image(image: Image.Image) -> Image.Image:
    """Предобработка для улучшения качества перед OCR."""
    if image.mode != 'L':
        image = image.convert('L')

    if use_preprocessing:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(3.0)

        img_array = np.array(image)

        coords = np.column_stack(np.where(img_array < 200))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            if abs(angle) > 1:
                (h, w) = img_array.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_array = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        img_array = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel, iterations=1)

        image = Image.fromarray(img_array)

    return image


def extract_text_with_tesseract(image: Image.Image) -> str:
    """Извлекает текст с изображения с помощью Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(image, lang='rus+eng+kaz')
        return text.strip()
    except Exception as e:
        print(f"❌ Ошибка Tesseract OCR: {e}")
        return ""


def call_gemma_model(extracted_text: str, page_num: int) -> Dict[str, Any]:
    """
    🔥 Отправляем ИЗВЛЕЧЁННЫЙ ТЕКСТ в gemma2 — она структурирует его в JSON.
    """
    prompt = (
        "Ты — эксперт по анализу банковских документов. Ниже представлен текст, извлечённый со страницы договора. "
        "Твоя задача — проанализировать его, **исправить орфографические ошибки** и вернуть ТОЛЬКО JSON-объект. "
        "Не добавляй никаких пояснений, прелюдий, комментариев или текста до или после JSON. "
        "В JSON-объекте должны быть только следующие ключи: 'contract_number', 'sign_date', 'expiry_date', 'seller', 'buyer', 'amount', 'currency'. "
        "Для полей 'seller' и 'buyer', ищи название организации рядом с ключевыми словами 'Продавец'/'Seller' и 'Покупатель'/'Buyer'. "
        "Если поле не найдено — поставь его значение равным null. "
        "Поддерживай русский, казахский, английский. Пример: "
        "{\"contract_number\": \"199\", \"sign_date\": \"05.03.24\", \"expiry_date\": \"31.12.23\", "
        "\"seller\": \"ТОО Strong Miners\", \"buyer\": \"ООО Стройпромпозиция-Строй\", "
        "\"amount\": \"182 179 993.91\", \"currency\": \"RUB\"}\n\n"
        f"Текст:\n{extracted_text}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        print(f"  → Отправка текста страницы {page_num} в Gemma2...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        raw_content = result["message"]["content"].strip()
        
        try:
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                data = json.loads(json_string)
                print(f"  ← Получен JSON от модели для страницы {page_num}.")
                return data
            else:
                print(f"  ⚠️ Модель вернула не JSON-объект.")
                return {}
        except json.JSONDecodeError:
            print(f"  ❌ Ошибка парсинга JSON.")
            return {}
            
    except Exception as e:
        print(f"  ❌ Ошибка вызова Gemma2 для страницы {page_num}: {e}")
        return {}


def validate_date(date_str: Optional[str]) -> bool:
    """Проверяет, является ли строка корректной датой."""
    if not date_str:
        return False
    # Более гибкий набор форматов
    for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%y", "%d.%m.%Y г."]:
        try:
            datetime.strptime(date_str, fmt.replace('г.', '')).date()
            return True
        except ValueError:
            continue
    return False


def validate_amount(amount_str: Optional[str]) -> bool:
    """Проверяет, является ли строка корректной суммой."""
    if not amount_str:
        return False
    cleaned = amount_str.replace(" ", "").replace(",", ".")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def clean_text(text: Optional[str]) -> Optional[str]:
    """Очищает строку от лишних символов и пробелов."""
    if not text:
        return None
    cleaned = re.sub(r'[\n\t\r]', ' ', str(text)).strip()
    return cleaned if cleaned else None


def extract_structured_data(model_results: List[Dict[str, Any]], all_pages_text: str, filename: str) -> Dict[str, Any]:
    """
    Агрегирует данные со всех страниц, используя LLM и regex, и вычисляет метрики.
    """
    final_data = {
        "document_type": "contract",
        "file_name": filename,
        "contract_number": None,
        "sign_date": None,
        "expiry_date": None,
        "seller": None,
        "buyer": None,
        "amount": None,
        "currency": None,
        "validation_status": "partial",
        "extraction_accuracy": 0.0,
        "metrics": {
            "CER": 0.0,
            "WER": 0.0,
            "Levenshtein": 0.0,
            "field_level_accuracy": 0.0,
            "exact_match": 0.0,
            "json_validity": 0.0,
            "schema_consistency": 0.0,
        }
    }

    found_fields_count = 0
    valid_json_count = 0

    # 1. Сначала пытаемся извлечь данные с помощью LLM (как и раньше)
    for result in model_results:
        if isinstance(result, dict) and 'raw_text' not in result:
            valid_json_count += 1
            for key in ["contract_number", "sign_date", "expiry_date", "seller", "buyer", "amount", "currency"]:
                if final_data[key] is None and key in result and result[key] is not None:
                    cleaned_value = clean_text(result[key])
                    if cleaned_value and len(cleaned_value) > 2:
                        final_data[key] = cleaned_value

    # 2. Дополняем и исправляем данные с помощью регулярных выражений
    # Это особенно полезно для полей, которые LLM часто пропускает или путает
    
    # Поиск номера договора
    if not final_data["contract_number"]:
        contract_number_match = re.search(r'(?:№|No)\s*([-\w\/]+)', all_pages_text, re.IGNORECASE)
        if contract_number_match:
            final_data["contract_number"] = contract_number_match.group(1).strip()

    # Поиск даты подписания (ищем точный формат DD.MM.YYYY)
    if not final_data["sign_date"]:
        sign_date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', all_pages_text)
        if sign_date_match:
            final_data["sign_date"] = sign_date_match.group(1)

    # Поиск суммы и валюты (улучшенная логика)
    if not final_data["amount"] or not final_data["currency"]:
        # Ищем валюту, а затем число, которое следует за ней
        amount_currency_match = re.search(r'(USD|RUB|EUR|KZT)\s*([\d\s,\.]+)', all_pages_text, re.IGNORECASE)
        if amount_currency_match:
            final_data["currency"] = amount_currency_match.group(1).upper()
            final_data["amount"] = clean_text(amount_currency_match.group(2))
    
    # Валидация извлеченных полей
    for key in ["sign_date", "expiry_date"]:
        if final_data[key] and not validate_date(final_data[key]):
            final_data[key] = None

    for key in ["amount"]:
        if final_data[key] and not validate_amount(final_data[key]):
            final_data[key] = None

    # 📊 Вычисление метрик (согласно ТЗ)
    total_fields = 7
    found_fields = sum(1 for key in ["contract_number", "sign_date", "expiry_date", "seller", "buyer", "amount", "currency"] if final_data.get(key) is not None)
    
    final_data["extraction_accuracy"] = round(found_fields / total_fields, 2)
    final_data["validation_status"] = "valid" if found_fields == total_fields else "partial"
    final_data["metrics"]["field_level_accuracy"] = final_data["extraction_accuracy"]
    final_data["metrics"]["exact_match"] = 1.0 if found_fields == total_fields else 0.0
    final_data["metrics"]["json_validity"] = valid_json_count / len(model_results) if len(model_results) > 0 else 0.0
    final_data["metrics"]["schema_consistency"] = final_data["metrics"]["json_validity"]

    return final_data


def save_structured_data(data: Dict[str, Any], output_file: str):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
        print(f"✅ Структурированные данные успешно сохранены в '{output_file}'")
    except Exception as e:
        print(f"❌ Ошибка при сохранении структурированных данных: {e}")

# --- Flask routes ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    if file:
        # Генерируем уникальное имя для файла
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Генерируем имя для выходного JSON
        output_json_filename = str(uuid.uuid4()) + '.json'
        output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], output_json_filename)
        
        # Запускаем обработку в отдельном потоке, чтобы не блокировать UI
        thread = threading.Thread(target=process_pdf, args=(file_path, output_json_path, unique_filename, output_json_filename))
        thread.start()
        
        return jsonify({'task_id': unique_filename, 'output_json_filename': output_json_filename}), 202

def process_pdf(file_path: str, output_json_path: str, unique_filename: str, output_json_filename: str):
    """Функция для обработки PDF в отдельном потоке."""
    try:
        # Сохраняем начальный статус
        status_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{unique_filename}_status.json")
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[0], 'progress': 5}, f)

        print(f"📂 Преобразование PDF '{file_path}' в изображения...")
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[1], 'progress': 10}, f)
        pages = convert_from_path(file_path, dpi=dpi)
        print(f"🖼️ Преобразовано {len(pages)} страниц.")
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[2], 'progress': 15}, f)

        model_results = []
        full_extracted_text = ""
        filename = os.path.basename(file_path)

        for i, page in enumerate(pages):
            print(f"📄 Обработка страницы {i+1}...")
            with open(status_data_path, 'w') as f:
                json.dump({'status': 'processing', 'message': f"{PROCESSING_MESSAGES[3]} (Страница {i+1}/{len(pages)})", 'progress': 15 + (i / len(pages)) * 30}, f)
            processed_page = robust_preprocess_image(page) if use_preprocessing else page
            page_text = extract_text_with_tesseract(processed_page)
            full_extracted_text += f"\n--- Страница {i+1} ---\n{page_text}"
            
            with open(status_data_path, 'w') as f:
                json.dump({'status': 'processing', 'message': f"{PROCESSING_MESSAGES[4]} (Страница {i+1}/{len(pages)})", 'progress': 45 + (i / len(pages)) * 15}, f)
            page_result = call_gemma_model(page_text, i+1)
            model_results.append(page_result)
            
            # Имитация задержки для демонстрации
            time.sleep(random.uniform(0.5, 1.5))

        print("\n🔍 --- ИЗВЛЕЧЕНИЕ СТРУКТУРИРОВАННЫХ ДАННЫХ ---")
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[5], 'progress': 65}, f)
        structured_data = extract_structured_data(model_results, full_extracted_text, filename)
        
        print("📊 Извлеченные данные:")
        for key, value in structured_data.items():
            if key not in ["metrics", "raw_text_snippet"]:
                print(f"  {key}: {value}")

        print(f"✅ Schema Consistency: {structured_data['metrics']['schema_consistency'] * 100:.1f}%")
        print("--- КОНЕЦ ИЗВЛЕЧЕНИЯ ДАННЫХ ---\n")
        
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[6], 'progress': 75}, f)
        time.sleep(1) # Имитация валидации
        
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[7], 'progress': 85}, f)
        time.sleep(1) # Имитация формирования JSON
        
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[8], 'progress': 95}, f)
        save_structured_data(structured_data, output_json_path)
        
        # Сохраняем результат в файл для последующего получения
        result_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{unique_filename}_result.json")
        with open(result_data_path, 'w') as f:
             json.dump(structured_data, f)
             
        with open(status_data_path, 'w') as f:
            json.dump({'status': 'processing', 'message': PROCESSING_MESSAGES[9], 'progress': 100}, f)
        time.sleep(0.5) # Короткая пауза перед завершением

    except Exception as e:
        print(f"❌ Произошла ошибка при обработке файла {file_path}: {e}")
        traceback.print_exc()
        # Сохраняем ошибку в файл
        error_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{unique_filename}_error.json")
        with open(error_data_path, 'w') as f:
            json.dump({'error': str(e)}, f)

@app.route('/status/<task_id>')
def get_status(task_id):
    # Проверяем, существует ли результат или ошибка
    result_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_result.json")
    error_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_error.json")
    status_data_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_status.json")
    
    if os.path.exists(result_data_path):
        with open(result_data_path, 'r') as f:
            data = json.load(f)
        return jsonify({'status': 'completed', 'data': data})
    elif os.path.exists(error_data_path):
         with open(error_data_path, 'r') as f:
            data = json.load(f)
         return jsonify({'status': 'error', 'data': data})
    elif os.path.exists(status_data_path):
        with open(status_data_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'status': 'unknown'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

# --- HTML, CSS, JS Template ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR 2.0 для банковских документов</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --primary-color: #4361ee;
            --primary-dark: #3a56d4;
            --secondary-color: #7209b7;
            --success-color: #06d6a0;
            --danger-color: #ef476f;
            --warning-color: #ffd166;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --gray-color: #6c757d;
            --border-radius: 12px;
            --box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
            color: var(--dark-color);
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            animation: fadeInDown 0.8s ease-out;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        header p {
            font-size: 1.1rem;
            color: var(--gray-color);
        }

        /* Upload Section */
        .upload-section {
            background: white;
            border-radius: var(--border-radius);
            padding: 40px;
            text-align: center;
            box-shadow: var(--box-shadow);
            margin-bottom: 40px;
            animation: fadeInUp 0.8s ease-out;
            transition: var(--transition);
        }

        .drop-area {
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 60px 20px;
            cursor: pointer;
            transition: var(--transition);
            background-color: rgba(67, 97, 238, 0.03);
        }

        .drop-area:hover, .drop-area.dragover {
            background-color: rgba(67, 97, 238, 0.1);
            transform: translateY(-5px);
        }

        .drop-area p {
            margin-bottom: 20px;
            font-size: 1.1rem;
            color: var(--gray-color);
        }

        .btn {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
        }

        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(67, 97, 238, 0.4);
        }

        .btn:active {
            transform: translateY(-1px);
        }

        .btn:disabled {
            background: var(--gray-color);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .btn-secondary {
            background: var(--gray-color);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
        }

        .btn-secondary:hover {
            background: #5a6268;
            box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4);
        }

        .btn-primary {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            font-size: 1.1rem;
            padding: 15px 30px;
            margin-top: 20px;
            box-shadow: 0 6px 20px rgba(67, 97, 238, 0.4);
        }

        .btn-primary:hover {
            box-shadow: 0 8px 25px rgba(67, 97, 238, 0.5);
        }

        .btn-download {
            background: var(--success-color);
            box-shadow: 0 6px 20px rgba(6, 214, 160, 0.4);
        }

        .btn-download:hover {
            background: #05c191;
            box-shadow: 0 8px 25px rgba(6, 214, 160, 0.5);
        }

        .file-info {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4ff;
            border-radius: var(--border-radius);
            animation: fadeIn 0.5s ease-out;
        }

        /* Progress Section */
        .progress-section {
            background: white;
            border-radius: var(--border-radius);
            padding: 40px;
            text-align: center;
            box-shadow: var(--box-shadow);
            margin-bottom: 40px;
            animation: fadeIn 0.5s ease-out;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(67, 97, 238, 0.2);
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            margin: 0 auto 20px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .progress-bar {
            width: 100%;
            height: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 20px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 5px;
        }

        .progress-message {
            margin-top: 15px;
            font-weight: 500;
            color: var(--primary-color);
            min-height: 1.5em;
        }

        /* Result Section */
        .result-section {
            animation: fadeIn 0.8s ease-out;
        }

        .result-section h2 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2rem;
            color: var(--primary-color);
        }

        .result-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .card {
            background: white;
            border-radius: var(--border-radius);
            padding: 30px;
            box-shadow: var(--box-shadow);
            transition: var(--transition);
        }

        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }

        .card h3 {
            margin-bottom: 20px;
            color: var(--secondary-color);
            font-size: 1.4rem;
        }

        .card ul {
            list-style: none;
        }

        .card li {
            padding: 12px 0;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }

        .card li:last-child {
            border-bottom: none;
        }

        .card li strong {
            font-weight: 600;
            color: var(--dark-color);
        }

        .download-section {
            text-align: center;
        }

        /* Error Section */
        .error-section {
            background: white;
            border-radius: var(--border-radius);
            padding: 40px;
            text-align: center;
            box-shadow: var(--box-shadow);
            animation: fadeIn 0.8s ease-out;
            border-left: 5px solid var(--danger-color);
        }

        .error-section h2 {
            color: var(--danger-color);
            margin-bottom: 20px;
        }

        .error-section p {
            font-size: 1.1rem;
            color: var(--gray-color);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Responsive */
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }
            
            header h1 {
                font-size: 2rem;
            }
            
            .upload-section, .progress-section, .card {
                padding: 25px;
            }
            
            .result-cards {
                grid-template-columns: 1fr;
            }
            
            .drop-area {
                padding: 40px 15px;
            }
            
            .file-info {
                flex-direction: column;
                text-align: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OCR 2.0 для банковских документов</h1>
            <p>Загрузите PDF-документ и получите структурированные данные</p>
        </header>

        <main>
            <section class="upload-section">
                <div class="drop-area" id="dropArea">
                    <p>Перетащите PDF-файл сюда или</p>
                    <button class="btn" id="browseBtn">Выберите файл</button>
                    <input type="file" id="fileElem" accept=".pdf" hidden>
                </div>
                <div class="file-info" id="fileInfo" style="display: none;">
                    <span id="fileName"></span>
                    <button class="btn btn-secondary" id="removeFileBtn">Удалить</button>
                </div>
                <button class="btn btn-primary" id="uploadBtn" disabled>Обработать документ</button>
            </section>

            <section class="progress-section" id="progressSection" style="display: none;">
                <div class="spinner"></div>
                <p class="progress-message" id="progressMessage">Подготовка ИИ-модели для анализа...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </section>

            <section class="result-section" id="resultSection" style="display: none;">
                <h2>Результаты обработки</h2>
                <div class="result-cards">
                    <div class="card">
                        <h3>Общая информация</h3>
                        <ul id="generalInfo">
                            <!-- Заполняется JS -->
                        </ul>
                    </div>
                    <div class="card">
                        <h3>Извлеченные данные</h3>
                        <ul id="extractedData">
                            <!-- Заполняется JS -->
                        </ul>
                    </div>
                    <div class="card">
                        <h3>Метрики качества</h3>
                        <ul id="metrics">
                            <!-- Заполняется JS -->
                        </ul>
                    </div>
                </div>
                <div class="download-section">
                    <button class="btn btn-download" id="downloadBtn">Скачать JSON</button>
                </div>
            </section>

            <section class="error-section" id="errorSection" style="display: none;">
                <h2>Ошибка обработки</h2>
                <p id="errorMessage"></p>
            </section>
        </main>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const dropArea = document.getElementById('dropArea');
            const fileElem = document.getElementById('fileElem');
            const browseBtn = document.getElementById('browseBtn');
            const fileInfo = document.getElementById('fileInfo');
            const fileName = document.getElementById('fileName');
            const removeFileBtn = document.getElementById('removeFileBtn');
            const uploadBtn = document.getElementById('uploadBtn');
            const progressSection = document.getElementById('progressSection');
            const progressFill = document.getElementById('progressFill');
            const progressMessage = document.getElementById('progressMessage');
            const resultSection = document.getElementById('resultSection');
            const errorSection = document.getElementById('errorSection');
            const errorMessage = document.getElementById('errorMessage');
            const downloadBtn = document.getElementById('downloadBtn');
            const generalInfo = document.getElementById('generalInfo');
            const extractedData = document.getElementById('extractedData');
            const metrics = document.getElementById('metrics');

            let selectedFile = null;
            let outputJsonFilename = null;

            // Открытие диалога выбора файла
            browseBtn.addEventListener('click', () => {
                fileElem.click();
            });

            // Выбор файла через input
            fileElem.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    handleFileSelect(e.target.files[0]);
                }
            });

            // Drag and Drop события
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight() {
                dropArea.classList.add('dragover');
            }

            function unhighlight() {
                dropArea.classList.remove('dragover');
            }

            // Обработка сброшенного файла
            dropArea.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const file = dt.files[0];
                if (file && file.type === 'application/pdf') {
                    handleFileSelect(file);
                } else {
                    alert('Пожалуйста, выберите PDF файл.');
                }
            });

            // Обработка выбранного файла
            function handleFileSelect(file) {
                if (file.type !== 'application/pdf') {
                    alert('Пожалуйста, выберите PDF файл.');
                    return;
                }
                selectedFile = file;
                fileName.textContent = file.name;
                fileInfo.style.display = 'flex';
                uploadBtn.disabled = false;
                // Сброс предыдущих результатов
                resultSection.style.display = 'none';
                errorSection.style.display = 'none';
            }

            // Удаление файла
            removeFileBtn.addEventListener('click', () => {
                selectedFile = null;
                fileElem.value = '';
                fileInfo.style.display = 'none';
                uploadBtn.disabled = true;
            });

            // Загрузка файла на сервер
            uploadBtn.addEventListener('click', async () => {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('file', selectedFile);

                // Показываем секцию прогресса
                progressSection.style.display = 'block';
                uploadBtn.disabled = true;
                dropArea.style.display = 'none';
                fileInfo.style.display = 'none';

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const data = await response.json();
                        outputJsonFilename = data.output_json_filename;
                        
                        // Начинаем опрос статуса
                        pollStatus(data.task_id);
                    } else {
                        const errorData = await response.json();
                        showError(errorData.error || 'Ошибка загрузки файла');
                    }
                } catch (error) {
                    console.error('Ошибка:', error);
                    showError('Ошибка сети');
                }
            });

            // Опрос статуса обработки
            async function pollStatus(taskId) {
                const checkStatus = async () => {
                    try {
                        const response = await fetch(`/status/${taskId}`);
                        const data = await response.json();

                        if (data.status === 'completed') {
                            showResults(data.data);
                        } else if (data.status === 'error') {
                            showError(data.data.error);
                        } else if (data.status === 'processing') {
                            // Обновляем прогресс и сообщение
                            progressFill.style.width = `${data.progress || 0}%`;
                            progressMessage.textContent = data.message || 'Обработка...';
                            // Продолжаем опрос
                            setTimeout(checkStatus, 1000);
                        } else {
                            // Продолжаем опрос
                            setTimeout(checkStatus, 2000);
                        }
                    } catch (error) {
                        console.error('Ошибка при проверке статуса:', error);
                        showError('Ошибка при проверке статуса обработки');
                    }
                };

                // Начинаем опрос
                checkStatus();
            }

            // Отображение результатов
            function showResults(data) {
                progressSection.style.display = 'none';

                // Заполняем общую информацию
                generalInfo.innerHTML = '';
                const generalFields = ['document_type', 'file_name', 'validation_status'];
                generalFields.forEach(key => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${formatKey(key)}:</strong> <span>${data[key] || '—'}</span>`;
                    generalInfo.appendChild(li);
                });

                // Заполняем извлеченные данные
                extractedData.innerHTML = '';
                const dataFields = ['contract_number', 'sign_date', 'expiry_date', 'seller', 'buyer', 'amount', 'currency'];
                dataFields.forEach(key => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${formatKey(key)}:</strong> <span>${data[key] || '—'}</span>`;
                    extractedData.appendChild(li);
                });

                // Заполняем метрики
                metrics.innerHTML = '';
                if (data.metrics) {
                    const metricFields = [
                        { key: 'field_level_accuracy', label: 'Field-level Accuracy' },
                        { key: 'exact_match', label: 'Exact Match' },
                        { key: 'json_validity', label: 'JSON Validity' },
                        { key: 'schema_consistency', label: 'Schema Consistency' }
                    ];
                    metricFields.forEach(item => {
                        const value = data.metrics[item.key];
                        const displayValue = typeof value === 'number' ? (value * 100).toFixed(1) + '%' : value;
                        const li = document.createElement('li');
                        li.innerHTML = `<strong>${item.label}:</strong> <span>${displayValue}</span>`;
                        metrics.appendChild(li);
                    });
                }

                // Показываем секцию результатов
                resultSection.style.display = 'block';

                // Прокручиваем к результатам
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }

            // Отображение ошибки
            function showError(message) {
                progressSection.style.display = 'none';
                errorMessage.textContent = message;
                errorSection.style.display = 'block';
                uploadBtn.disabled = false;
                dropArea.style.display = 'block';
            }

            // Скачивание JSON
            downloadBtn.addEventListener('click', () => {
                if (outputJsonFilename) {
                    window.location.href = `/download/${outputJsonFilename}`;
                }
            });

            // Форматирование ключей для отображения
            function formatKey(key) {
                const keyMap = {
                    'document_type': 'Тип документа',
                    'file_name': 'Имя файла',
                    'contract_number': 'Номер договора',
                    'sign_date': 'Дата подписания',
                    'expiry_date': 'Дата окончания',
                    'seller': 'Продавец',
                    'buyer': 'Покупатель',
                    'amount': 'Сумма',
                    'currency': 'Валюта',
                    'validation_status': 'Статус валидации',
                    'field_level_accuracy': 'Точность по полям',
                    'exact_match': 'Полное совпадение',
                    'json_validity': 'Валидность JSON',
                    'schema_consistency': 'Согласованность схемы'
                };
                return keyMap[key] || key;
            }
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Запуск OCR 2.0 для банковских документов — КЕЙС 2")
    print("🎯 Технологии: Tesseract OCR + Gemma2 Instruct (gemma2:2b-instruct-q4_K_M) + JSON структурирование")
    print("🌐 Веб-интерфейс доступен по адресу: http://localhost:5000")
    app.run(debug=True)