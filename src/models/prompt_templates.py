from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from src.utils.data_loader import DataLoader
from src.models.example_selectors import *

DATA_FILE = '../data/outcome_similarity/train.tsv'


data_loader = DataLoader(DATA_FILE)
examples = data_loader.load_dict()


# Zero shot
# TODO Evaluated: Are the following sentences semantically similar? Answer with either Yes or No.
sentence_template = PromptTemplate.from_template("""Are the sentences semantically similar? Answer with either Yes or No.

First sentence: {out1}

Second sentence: {out2}

Answer:""")


outcome_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


# TODO Evaluated: Is the following primary outcome correctly reported? Answer with either Yes or No.
role_template = PromptTemplate.from_template("""You are a clinical trial report reviewer.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


# TODO Evaluated: One type of incorrect reporting is changing the predefined primary outcome of a clinical trial. That way the researchers can report only the outcomes that support their hypothesis.
# TODO Evaluated: Is the following primary outcome correctly reported? Answer with either Yes or No.
wikipedia_definition_template = PromptTemplate.from_template("""You are a clinical trial report reviewer.

---

Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way, researchers can cherry pick an outcome which is statistically significant.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


koroleva_definition_template = PromptTemplate.from_template("""You are a clinical trial report reviewer.

---

Outcome switching is defined as unjustified change of the predefined trial outcomes. This leads to researchers reporting only the outcomes that support their hypothesis.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


step_template = PromptTemplate.from_template("""You are a clinical trial report reviewer.

---

Is the following primary outcome correctly reported? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:""")


# Few shot
prefix = """You are a clinical trial report reviewer."""


# TODO Evaluated: Is the following primary outcome correctly reported? Answer with either Yes or No.
example_template = PromptTemplate.from_template("""Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer: {answer}""")


example_separator = """

---

"""


# TODO Evaluated: Is the following primary outcome correctly reported? Answer with either Yes or No.
suffix = """Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

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
