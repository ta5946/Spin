from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from src.utils.data_loader import DataLoader
from src.models.example_selectors import *

DATA_FILE = '../data/outcome_similarity/train.tsv'


data_loader = DataLoader(DATA_FILE)
examples = data_loader.load_dict()


# Zero shot
sentence_template = PromptTemplate.from_template("""Are the following sentences semantically similar? Answer with either Yes or No.

First sentence: {out1}

Second sentence: {out2}

Answer:""")


outcome_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


outcome_definition_template = PromptTemplate.from_template("""Outcome switching is unjustified change of the predefined trial outcomes, leading to reporting only the favourable outcomes that support the hypothesis of the researchers.

Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


outcome_step_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:""")


# Few shot
prefix = """Outcome switching is unjustified change of the predefined trial outcomes, leading to reporting only the favourable outcomes that support the hypothesis of the researchers."""


example_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer: {answer}""")


example_separator = """

---

"""


suffix = """Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:"""


random_template = FewShotPromptTemplate(
    example_selector=RandomExampleSelector(examples, 1),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


negative_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [0]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


positive_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [1]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


negative_positive_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [0, 1]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


positive_negative_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [1, 0]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)
