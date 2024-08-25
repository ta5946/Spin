import random
from langchain_core.example_selectors.base import BaseExampleSelector


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples, n_choices):
        self.examples = examples
        self.n_choices = n_choices

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        examples = random.choices(self.examples, k=self.n_choices)
        return examples


class LabelExampleSelector(BaseExampleSelector):
    def __init__(self, examples, labels):
        self.negative_examples = [example for example in examples if example['label'] == 0]
        self.positive_examples = [example for example in examples if example['label'] == 1]
        self.labels = labels

    def add_example(self, example):
        if example['label'] == 0:
            self.negative_examples.append(example)
        else:
            self.positive_examples.append(example)

    def select_examples(self, input_variables):
        examples = []
        for label in self.labels:
            if label == 0:
                examples.append(random.choice(self.negative_examples))
            else:
                examples.append(random.choice(self.positive_examples))
        return examples
