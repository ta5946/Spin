import random
from langchain_core.example_selectors.base import BaseExampleSelector


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        example = random.choice(self.examples)
        return [example]


class NegativeExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.negative_examples = [example for example in examples if example['label'] == 'No']

    def add_example(self, example):
        if example['label'] == 'No':
            self.negative_examples.append(example)

    def select_examples(self, input_variables):
        example = random.choice(self.negative_examples)
        return [example]


class PositiveExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.positive_examples = [example for example in examples if example['label'] == 'Yes']

    def add_example(self, example):
        if example['label'] == 'Yes':
            self.positive_examples.append(example)

    def select_examples(self, input_variables):
        example = random.choice(self.positive_examples)
        return [example]
