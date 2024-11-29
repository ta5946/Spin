from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.utils.data_loader import DataLoader
from src.models.example_selectors import *

DATA_FILE = '../data/dev/train.tsv'


data_loader = DataLoader(DATA_FILE)
examples = data_loader.load_dict()
str_examples = data_loader.load_dict(str_only=True)

sentence_transformers_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# Zero shot
sentence_template = PromptTemplate.from_template("""Are the sentences semantically similar? Answer with either Yes or No.

First sentence: {out1}

Second sentence: {out2}

Answer:""")


outcome_template = PromptTemplate.from_template("""Does the reported outcome match the defined primary outcome? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


role_template = PromptTemplate.from_template("""You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


detail_template = PromptTemplate.from_template("""You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Primary outcome is defined at the start of a clinical trial and is the observed variable of the study.
This outcome is later reported in the results section of the report.
The reported outcome should match the defined primary outcome.
That means it should include all its components and details.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


article_definition_template = PromptTemplate.from_template("""You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Outcome switching is an unjustified change of the predefined trial outcomes, leading to reporting only the favourable outcomes that support the hypothesis of the researchers. Outcome switching is one of the most common types of spin. It can consist in omitting the primary outcome in the results and conclusions of the abstract, or in the focus on significant secondary outcomes.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


wikipedia_definition_template = PromptTemplate.from_template("""You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way researchers can cherry pick an outcome which is statistically significant.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:""")


chain_of_thought_template = PromptTemplate.from_template("""You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Does the reported outcome match the defined primary outcome? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:""")


# Few shot
prefix = """You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes."""


example_template = PromptTemplate.from_template("""Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer: {ans}""")


example_separator = """

---

"""


suffix = """Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:"""


random_example_template = FewShotPromptTemplate(
    example_selector=RandomExampleSelector(examples, 1),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


negative_example_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [0]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


positive_example_template = FewShotPromptTemplate(
    example_selector=LabelExampleSelector(examples, [1]),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


similar_example_template = FewShotPromptTemplate(
    example_selector=SemanticSimilarityExampleSelector.from_examples(str_examples, sentence_transformers_embeddings, Chroma, 1),
    prefix=prefix,
    example_prompt=example_template,
    example_separator=example_separator,
    suffix=suffix,
    input_variables=['out1', 'out2']
)


explanation_template = """Why? Explain your answer in one sentence.
            
Explanation:"""
