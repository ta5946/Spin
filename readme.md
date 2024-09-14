# Detecting spin in clinical trials with large language models


## Language models

### OLMo-7B-Instruct
- Description: **Open training** and weights general purpose model.
- Size: 6.9B parameters
- Release date: February 2024
- Link: [https://huggingface.co/allenai/OLMo-7B-Instruct-hf](https://huggingface.co/allenai/OLMo-7B-Instruct-hf)

### Mistral-7B-Instruct-v0.2
- Description: Open weights general purpose model.
- Size: 7.2B parameters
- Release date: December 2023
- Link: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

### BioMistral-7B-DARE
- Description: Open weights medical **domain adapted** model.
- Size: 7.2B parameters
- Release date: February 2024
- Link: [https://huggingface.co/BioMistral/BioMistral-7B-DARE](https://huggingface.co/BioMistral/BioMistral-7B-DARE)

### Meta-Llama-3-8B-Instruct
- Description: **State of the art** open weights model.
- Size: 8B parameters
- Release date: April 2024
- Link: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)


## Dataset

- Labeled pairs of primary and reported outcomes of PubMed clinical trial abstracts.
- 100 pairs for few shot examples and threshold tuning.
- 2600 pairs for prompt tuning and model evaluation.
- Biased towards **negative examples**, where the reported outcome does not match the primary outcome.

![docs/outcome_similarity/distribution.png](docs/outcome_similarity/distribution.png)


## Prompt templates

### Zero shot

- Sentence template:
```text
<|endoftext|><|user|>
Are the sentences semantically similar? Answer with either Yes or No.

First sentence: {out1}

Second sentence: {out2}

Answer:
<|assistant|>
```

- Outcome template:
```text
<|endoftext|><|user|>
Does the reported outcome match the defined primary outcome? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```

- Role template:
```text
<|endoftext|><|user|>
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.
                                                
Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```

- Article definition template:
```text
<|endoftext|><|user|>
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Outcome switching is an unjustified change of the predefined trial outcomes, leading to reporting only the favourable outcomes that support the hypothesis of the researchers. Outcome switching is one of the most common types of spin. It can consist in omitting the primary outcome in the results and conclusions of the abstract, or in the focus on significant secondary outcomes.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```

- Wikipedia definition template:
```text
<|endoftext|><|user|>
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Outcome switching is the practice of changing the primary or secondary outcomes of a clinical trial after its initiation. An outcome is the goal of the clinical trial, such as survival after five years for cancer treatment. Outcome switching can lead to bias and undermine the reliability of the trial, for instance when outcomes are switched after researchers already have access to trial data. That way, researchers can cherry pick an outcome which is statistically significant.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```

- Chain of thought template:
```text
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Does the reported outcome match the defined primary outcome? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:
<|assistant|>
```

### One shot

- Random and similar example template:
```text
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {example_out1}

Reported outcome: {example_out2}

Answer: {example_answer}

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```


## Results

### Baseline models

| Model                                | Auc score |
|--------------------------------------|-----------|
| Constant classifier                  | 0.500     |
| Random classifier                    | 0.417     |
| Lemmas classifier                    | 0.831     |
| **Stems classifier**                 | **0.871** |
| Levenshtein classifier               | 0.801     |
| Sequence classifier                  | 0.850     |
| Spacy classifier                     | 0.711     |
| Word2Vec classifier                  | 0.814     |
| SciBERT classifier                   | 0.782     |
| **Sentence transformers classifier** | **0.884** |

### Large language models

| Auc score                     | Olmo  | Mistral | BioMistral | Llama     |
|-------------------------------|-------|---------|------------|-----------|
| Sentence template             | 0.897 | 0.878   | **0.893**  | 0.875     |
| Outcome template              | 0.922 | 0.908   | **0.931**  | 0.939     |
| Role template                 | 0.912 | 0.906   | 0.919      | 0.920     |
| Article definition template   | 0.897 | 0.922   | 0.907      | **0.931** |
| Wikipedia definition template | 0.884 | 0.908   | 0.908      | **0.934** |
| Random example template       | 0.907 | 0.892   | 0.881      | 0.860     |
| Similar example template      | 0.880 | 0.874   | 0.903      | 0.823     |

> **Note**: Model role is a part of the user prompt, except in the case of Llama that uses a dedicated system prompt.

Key takeaways:
- Domain specific BioMistral is most effective with minimal prompt engineering.
- State of the art Llama 3 excels at understanding complex prompts.
- Olmo and Mistral are relatively stable in their predictions.


## References

- [Measuring semantic similarity of clinical trial outcomes using deep pre-trained language representations](https://www.sciencedirect.com/science/article/pii/S2590177X19300575)
- [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676)
- [A Comprehensive Overview of Large Language Models](https://arxiv.org/abs/2307.06435)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- [Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review](https://arxiv.org/abs/2310.14735)
- [Large Language Models are Few-Shot Clinical Information Extractors](https://arxiv.org/abs/2205.12689)
- [Analyzing Spin in Abstracts of Orthopaedic Randomized Controlled Trials With Statistically Insignificant Primary Endpoints](https://www.sciencedirect.com/science/article/pii/S0749806320300384?casa_token=646vq-iv8coAAAAA:oVcdcHfVD1GqLqt68JN8XucIxGzKpOyq03dT8xG1sgAV6GIRQgVEQFmyd4O4r5JXHz8ZCenAoQWW)
- [BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains](https://arxiv.org/abs/2402.10373)
- [COMPare: A prospective cohort study correcting and monitoring 58 misreported trials in real time](https://link.springer.com/article/10.1186/s13063-019-3173-2)
