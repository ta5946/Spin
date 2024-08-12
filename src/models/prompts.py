from langchain_core.prompts import PromptTemplate


similarity_template = PromptTemplate.from_template("""Are the given sentences semantically similar? Answer with either Yes or No.

First sentence: {sentence2}
Second sentence: {sentence1}""")
