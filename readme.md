# Detecting spin in clinical trials with large language models

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

- Chain of thought template:
```text
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Is the following primary outcome correctly reported? Lets think step by step.

Primary outcome: {out1}

Reported outcome: {out2}

Steps:
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

- Article definition template:
```text
<|endoftext|><|user|>
You are a clinical trial report reviewer. Your task is to detect incorrectly reported outcomes.

---

Outcome switching is defined as unjustified change of the predefined trial outcomes. This leads to researchers reporting only the outcomes that support their hypothesis.

---

Does the reported outcome match the defined primary outcome? Answer with either Yes or No.

Primary outcome: {out1}

Reported outcome: {out2}

Answer:
<|assistant|>
```

### One shot

- Random template:
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
