from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator
from src.utils.experiment import Experiment
from src.models.base_models import *
from src.models.language_models import *
from src.models.prompt_templates import *

DATA_FILE = '../data/outcome_similarity/train.tsv'

# TODO
EXPERIMENT_MODEL = Probability(random_template)
EXPERIMENT_NAME = 'Probability(random_template)'


data_loader = DataLoader(DATA_FILE)
data = data_loader.load_df()

model = EXPERIMENT_MODEL
model_evaluator = ModelEvaluator(model, data)

experiment = Experiment(model_evaluator, EXPERIMENT_NAME)
experiment.run()
