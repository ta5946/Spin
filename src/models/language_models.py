import torch
import weave
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax


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


class OlmoText:
    def __init__(self, template):
        self.model = Olmo()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction


class OlmoProbability:
    def __init__(self, template, threshold=0.8):
        self.model = Olmo()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction


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
        return prediction, prediction


class MistralProbability:
    def __init__(self, template, threshold=0.5):
        self.model = Mistral()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction


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
        return prediction, prediction


class BioProbability:
    def __init__(self, template, threshold=0.5):
        self.model = Bio()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction


class Llama:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('test')

    @weave.op
    def generate_text(self, user_text, n_tokens=1000):
        input_chat = [
            {'role': 'user', 'content': user_text},
        ]
        inputs = self.tokenizer.apply_chat_template(input_chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        n_inputs = inputs.shape[1]

        outputs = self.model.generate(inputs, max_new_tokens=n_tokens)
        outputs = outputs[0, n_inputs:]
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=False)
        return generated_text

    @weave.op
    def generate_probability(self, user_text):
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
        return prediction, prediction


class LlamaProbability:
    def __init__(self, template, threshold=0.5):
        self.model = Llama()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction
