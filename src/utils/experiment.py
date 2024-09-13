import wandb


class Experiment:
    def __init__(self, evaluator, name=''):
        self.evaluator = evaluator
        self.tracking = bool(name)

        if self.tracking:
            wandb.login()
            wandb.init(project='test', name=name)

    def run(self):
        self.evaluator.evaluate()
        metrics = self.evaluator.get_metrics()
        print(metrics)

        if self.tracking:
            wandb.log(metrics)
            wandb.finish()
