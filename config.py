import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11m_robot_detect_v2.engine')
DISTANCES_PATH = os.path.join(BASE_DIR, 'data', 'distances.json')
CROP_PATH = os.path.join(BASE_DIR, 'data', 'crop_calibration.json')

# Mapping from model class id to label.
CLASS_MAPPING = {
    0: 'robot',
    1: 'wall_bottom',
    2: 'wall_corner',
    3: 'wall_left',
    4: 'wall_right',
    5: 'wall_top'
}
