from pathlib import Path

# Folder parameters
ROOT = Path('D:/Data')
SOURCE_FOLDER = ROOT
OUTPUT_FOLDER = ROOT
MODEL_FOLDER = ROOT / 'models' / 'yolov10'

# Source parameters
INPUT_VIDEO = 'C3_detenido.mp4'
OUTPUT_NAME = 'C3_detenido_yolov10'

# Deep Learning model configuration
MODEL_WEIGHTS = 'yolov10x.pt'

# Inference configuration
CLASS_FILTER = [0,1,2,3,5,7]
IMAGE_SIZE = 640
CONFIDENCE = 0.25
