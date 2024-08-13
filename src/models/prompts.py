from langchain_core.prompts import PromptTemplate


sentence_template = PromptTemplate.from_template("""Are the following sentences semantically similar? Answer with either Yes or No.

First sentence: {sentence1}

Second sentence: {sentence2}""")


outcome_template = PromptTemplate.from_template("""Is this primary outcome correctly reported? Answer with either Yes or No.
                                                
Primary outcome: {sentence1}

Reported outcome: {sentence2}""")


# TODO Define outcome_switching_template
