from utils.data_loader import DataLoader
from utils.model_evaluator import ModelEvaluator
from utils.experiment import Experiment
from models.base_models import *
from models.language_models import *
from models.prompts import *

DATA_FILE = '../data/outcome_similarity/train.tsv'

EXPERIMENT_MODEL = OlmoInstruct(similarity_template)
EXPERIMENT_NAME = 'OlmoInstruct(similarity_template)'


data_loader = DataLoader(DATA_FILE)
data = data_loader.load()

model = EXPERIMENT_MODEL
model_evaluator = ModelEvaluator(model, data)

experiment = Experiment(model_evaluator, EXPERIMENT_NAME)
experiment.run()
