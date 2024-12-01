from dotenv import load_dotenv
from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator
from src.utils.experiment import Experiment
from src.models.base_models import *
from src.models.language_models import *
from src.models.prompt_templates import *

load_dotenv()
OUT_FILE = ''
DATA_FILE = '../data/dev/test.tsv'

# TODO Evaluate models
EXPERIMENT_MODEL = OlmoText(similar_example_template)
EXPERIMENT_NAME = ''


data_loader = DataLoader(DATA_FILE)
data = data_loader.load_df()

model = EXPERIMENT_MODEL
model_evaluator = ModelEvaluator(model, data, OUT_FILE)

experiment = Experiment(model_evaluator, EXPERIMENT_NAME)
experiment.run()
