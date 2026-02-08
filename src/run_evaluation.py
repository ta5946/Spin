import os
import json
import wandb
from dotenv import load_dotenv
from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator
from src.models.base_models import *
from src.models.language_models import *
from src.models.prompt_templates import *

load_dotenv()
DATA_DIR = '../data/test/'

# TODO Evaluate models
EVALUATION_MODEL = GemmaProbability(detail_template)
MODEL_DIR = 'detail_template'
EVALUATION_NAME = ''


# files
train_file = os.path.join(DATA_DIR, 'train.tsv')
test_file = os.path.join(DATA_DIR, 'test.tsv')
out_dir = os.path.join(DATA_DIR, MODEL_DIR)
os.makedirs(out_dir, exist_ok=True)

predictions_file = os.path.join(out_dir, 'predictions.tsv')
train_metrics_file = os.path.join(out_dir, 'train_metrics.json')
test_metrics_file = os.path.join(out_dir, 'test_metrics.json')

# train
train_data_loader = DataLoader(train_file)
train_data = train_data_loader.load_df()
train_model = EVALUATION_MODEL
train_model_evaluator = ModelEvaluator(train_model, train_data)

train_model_evaluator.evaluate()
train_metrics = train_model_evaluator.get_metrics()
with open(train_metrics_file, 'w') as file:
    json.dump(train_metrics, file, indent=4)
    print(train_metrics)

test_threshold = train_metrics['j_threshold']

# test
test_data_loader = DataLoader(test_file)
test_data = test_data_loader.load_df()
test_model = EVALUATION_MODEL
test_model.threshold = test_threshold
test_model_evaluator = ModelEvaluator(test_model, test_data, predictions_file)

test_model_evaluator.evaluate()
test_metrics = test_model_evaluator.get_metrics()
with open(test_metrics_file, 'w') as file:
    json.dump(test_metrics, file, indent=4)
    print(test_metrics)

if EVALUATION_NAME:
    wandb.login()
    wandb.init(project='outcome_similarity_detection', name=EVALUATION_NAME)
    wandb.log(test_metrics)
