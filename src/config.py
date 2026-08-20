import os.path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEEKS_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "VK-LSVD", "subsamples", "up0.001_ip0.001")
WEEKS_TRAIN_DATA = os.path.join(WEEKS_DATA_ROOT, "train")
WEEKS_VALIDATE_DATA = os.path.join(WEEKS_DATA_ROOT, "validation")

META_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "VK-LSVD", "metadata")
