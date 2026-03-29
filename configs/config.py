"""
Конфигурационный файл для проекта CNN Picture Classification
"""

from pathlib import Path


class Config:
    """Конфигурация обучения модели"""
    
    # Пути
    DATA_DIR = Path("/kaggle/input/intel-image-classification")
    TRAIN_DIR = DATA_DIR / "seg_train" / "seg_train"
    TEST_DIR = DATA_DIR / "seg_test" / "seg_test"
    OUTPUT_DIR = Path("./outputs")
    
    # Параметры модели
    NUM_CLASSES = 6
    IMAGE_SIZE = (150, 150)
    
    # Параметры обучения
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    NUM_WORKERS = 2
    
    # Классы
    CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Device
    DEVICE = 'cuda'
    
    def __init__(self):
        """Инициализация конфигурации"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        return f"Config(batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE}, epochs={self.NUM_EPOCHS})"
