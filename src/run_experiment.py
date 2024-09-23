from dotenv import load_dotenv
from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator
from src.utils.experiment import Experiment
from src.models.base_models import *
from src.models.language_models import *
from src.models.prompt_templates import *

load_dotenv()
OUT_FILE = ''
DATA_FILE = '../data/outcome_similarity/val.tsv'

# TODO Evaluate models
EXPERIMENT_MODEL = LlamaProbability(role_template)
EXPERIMENT_NAME = ''


data_loader = DataLoader(DATA_FILE)
data = data_loader.load_df().sample(n=200, random_state=0)

model = EXPERIMENT_MODEL
model_evaluator = ModelEvaluator(model, data, OUT_FILE)

experiment = Experiment(model_evaluator, EXPERIMENT_NAME)
experiment.run()
