import os
import numpy as np
import torch
import weave
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_openai import ChatOpenAI
from torch.nn.functional import softmax


# TODO Merge generic architectures into one class with model path as parameter
class Olmo:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-7B-Instruct-hf')
        self.model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-7B-Instruct-hf', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        outputs = self.model.generate(inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        generated_scores = outputs.scores[0][-1]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores, dim=0)[1].item()
        return yes_probability

    @weave.op
    def generate_explanation(self, user_prediction_text, generated_prediction_text, user_explanation_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_prediction_text},
            {'role': 'assistant', 'content': generated_prediction_text},
            {'role': 'user', 'content': user_explanation_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text


class OlmoText:
    def __init__(self, template):
        self.model = Olmo()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction, ''


class OlmoProbability:
    def __init__(self, template, threshold=0.1):
        self.model = Olmo()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class OlmoExplanation:
    def __init__(self, prediction_template, explanation_template, threshold=0.1):
        self.model = Olmo()
        self.prediction_template = prediction_template
        self.explanation_template = explanation_template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_prediction_text = self.prediction_template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_prediction_text)

        prediction = int(score >= self.threshold)
        generated_prediction_text = 'No, the reported outcome does not match the primary outcome.' if prediction == 0 else 'Yes, the reported outcome matches the primary outcome.'
        explanation = self.model.generate_explanation(user_prediction_text, generated_prediction_text, self.explanation_template)
        return score, prediction, explanation


class Mistral:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
        self.model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        outputs = self.model.generate(inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        generated_scores = outputs.scores[0][-1]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores, dim=0)[1].item()
        return yes_probability


class MistralText:
    def __init__(self, template):
        self.model = Mistral()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction, ''


class MistralProbability:
    def __init__(self, template, threshold=0.8):
        self.model = Mistral()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Bio:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B-DARE')
        self.model = AutoModelForCausalLM.from_pretrained('BioMistral/BioMistral-7B-DARE', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        outputs = self.model.generate(inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        generated_scores = outputs.scores[0][-1]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores, dim=0)[1].item()
        return yes_probability


class BioText:
    def __init__(self, template):
        self.model = Bio()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction, ''


class BioProbability:
    def __init__(self, template, threshold=0.2):
        self.model = Bio()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Gemma:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('google/medgemma-4b-it')
        self.model = AutoModelForCausalLM.from_pretrained('google/medgemma-4b-it', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
        input_chat = [
            {'role': 'user', 'content': user_text}
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        outputs = self.model.generate(inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        generated_scores = outputs.scores[0][-1]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores, dim=0)[1].item()
        return yes_probability


class GemmaText:
    def __init__(self, template):
        self.model = Bio()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction, ''


class GemmaProbability:
    def __init__(self, template, threshold=0.2):
        self.model = Bio()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Llama:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        self.text_seperator = '---'
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        if self.text_seperator in user_text:
            system_text = user_text.split(self.text_seperator, 1)[0].strip()
            user_text = user_text.split(self.text_seperator, 1)[1].strip()
            input_chat = [
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ]
        else:
            input_chat = [
                {'role': 'user', 'content': user_text},
            ]

        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
        if self.text_seperator in user_text:
            system_text = user_text.split(self.text_seperator, 1)[0].strip()
            user_text = user_text.split(self.text_seperator, 1)[1].strip()
            input_chat = [
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ]
        else:
            input_chat = [
                {'role': 'user', 'content': user_text},
            ]

        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs)
        generated_scores = outputs.logits[0, -1, :]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores, dim=0)[1].item()
        return yes_probability


class LlamaText:
    def __init__(self, template):
        self.model = Llama()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction, ''


class LlamaProbability:
    def __init__(self, template, threshold=0.1):
        self.model = Llama()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Gpt:
    def __init__(self):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.model = ChatOpenAI(model='gpt-4-turbo-2024-04-09', api_key=OPENAI_API_KEY, temperature=0, max_tokens=1, logprobs=True, top_logprobs=20)
        self.no_token = 'No'
        self.yes_token = 'Yes'
        self.text_seperator = '---'
        weave.init('test')

    @weave.op
    def generate_probability(self, user_text):
        if self.text_seperator in user_text:
            system_text = user_text.split(self.text_seperator, 1)[0].strip()
            user_text = user_text.split(self.text_seperator, 1)[1].strip()
            input_chat = [
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ]
        else:
            input_chat = [
                {'role': 'user', 'content': user_text},
            ]

        outputs = self.model.invoke(input_chat)
        generated_logprobs = outputs.response_metadata['logprobs']['content'][0]['top_logprobs']
        no_logprob = None
        yes_logprob = None
        for logprob in generated_logprobs:
            if logprob['token'] == self.no_token:
                no_logprob = logprob['logprob']
            elif logprob['token'] == self.yes_token:
                yes_logprob = logprob['logprob']
        if no_logprob is not None and yes_logprob is not None:
            yes_probability = np.exp(yes_logprob)
            no_probability = np.exp(no_logprob)
            return yes_probability / (no_probability + yes_probability)
        elif no_logprob is None:
            return 1
        elif yes_logprob is None:
            return 0
        else:
            return 0.5


class GptProbability:
    def __init__(self, template, threshold=0.1):
        self.model = Gpt()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)
        print(score)

        prediction = int(score >= self.threshold)
        return score, prediction, ''
