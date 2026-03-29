# CNN Picture Classification

Классификация изображений с использованием сверточной нейронной сети (CNN) на PyTorch.

## 📋 Описание

Проект реализует классификацию изображений по 6 классам:
- 🏢 Buildings (Здания)
- 🌲 Forest (Лес)
- 🏔️ Glacier (Ледник)
- ⛰️ Mountain (Гора)
- 🌊 Sea (Море)
- 🛣️ Street (Улица)

## 🚀 Возможности

- ✅ Кастомная архитектура CNN с Batch Normalization и Dropout
- ✅ Data augmentation для улучшения обобщающей способности
- ✅ Визуализация истории обучения
- ✅ Метрики качества (Confusion Matrix, Precision, Recall, F1-Score)
- ✅ Визуализация активаций сверточных слоев
- ✅ Сохранение лучшей модели

## 📦 Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/CNN-Picture-Classification.git
cd CNN-Picture-Classification

# Установите зависимости
pip install -r requirements.txt

# Скачайте датасет (опционально)
# Датасет: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
