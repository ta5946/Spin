from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator
from src.utils.experiment import Experiment
from src.models.base_models import *
from src.models.language_models import *
from src.models.prompts import *

DATA_FILE = '../data/outcome_similarity/train.tsv'

# TODO Evaluate ZeroShot(sentence_template), ZeroShot(outcome_template), ZeroShotProb(sentence_template), ZeroShotProb(outcome_template)
EXPERIMENT_MODEL = SciBert()
EXPERIMENT_NAME = ''


data_loader = DataLoader(DATA_FILE)
data = data_loader.load()

model = EXPERIMENT_MODEL
model_evaluator = ModelEvaluator(model, data)

experiment = Experiment(model_evaluator, EXPERIMENT_NAME)
experiment.run()
