import os
from pathlib import Path
import dotenv
from dotenv import find_dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
BASE_PATH = Path(find_dotenv()).parent
DATASET_RAW = os.path.join(BASE_PATH, "data", "raw")