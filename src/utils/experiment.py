import wandb


class Experiment:
    def __init__(self, evaluator, title=''):
        self.evaluator = evaluator
        self.tracking = bool(title)

        if self.tracking:
            wandb.login()
            wandb.init(project='outcome_similarity_detection', name=title)

    def run(self):
        self.evaluator.evaluate()
        scores = self.evaluator.get_scores()
        print(scores)

        if self.tracking:
            wandb.log(scores)
            wandb.finish()
