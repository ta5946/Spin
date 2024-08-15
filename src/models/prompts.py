from langchain_core.prompts import PromptTemplate


sentence_template = PromptTemplate.from_template("""Are the following sentences semantically similar? Answer with either Yes or No.

First sentence: {sentence1}

Second sentence: {sentence2}""")


outcome_template = PromptTemplate.from_template("""Is the following primary outcome correctly reported? Answer with either Yes or No.
                                                
Primary outcome: {sentence1}

Reported outcome: {sentence2}""")

# TODO Is the following reported outcome consistent with the primary outcome? Answer with either Yes or No.
definition_template = PromptTemplate.from_template("""Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way, researchers can cherry pick an outcome which is statistically significant.

Is the following primary outcome correctly reported? Answer with either Yes or No.

Primary outcome: {sentence1}

Reported outcome: {sentence2}""")


chain_template = PromptTemplate.from_template("""Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way, researchers can cherry pick an outcome which is statistically significant.

Is the following reported outcome consistent with the primary outcome? Lets think step by step.

Primary outcome: {sentence1}

Reported outcome: {sentence2}""")
