from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from src.utils.data_loader import DataLoader
from src.models.example_selectors import *

DATA_FILE = '../data/outcome_similarity/train.tsv'


data_loader = DataLoader(DATA_FILE)
examples = data_loader.load_dict()


sentence_template = PromptTemplate.from_template("""Are the following sentences semantically similar? Answer with either Yes or No.

First sentence: {out1}

Second sentence: {out2}

Answer:""")


outcome_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


definition_template = PromptTemplate.from_template("""Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way, researchers can cherry pick an outcome which is statistically significant.

Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


# TODO
cot_template = PromptTemplate.from_template("""Outcome switching is unjustified change of the pre-defined trial outcomes, leading to reporting only the favourable outcomes that support the hypothesis of the researchers.

Is the following primary outcome correctly reported? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:""")


example_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer: {label}

---""")


suffix = """Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:"""


random_template = FewShotPromptTemplate(
    example_selector=RandomExampleSelector(examples),
    example_prompt=example_template,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


negative_template = FewShotPromptTemplate(
    example_selector=NegativeExampleSelector(examples),
    example_prompt=example_template,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


positive_template = FewShotPromptTemplate(
    example_selector=PositiveExampleSelector(examples),
    example_prompt=example_template,
    suffix=suffix,
    input_variables=['out1', 'out2']
)
