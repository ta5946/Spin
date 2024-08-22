import torch
import weave
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax

HF_MODEL = 'allenai/OLMo-7B-Instruct-hf'


class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map='cuda', torch_dtype=torch.bfloat16)
        self.no_token = self.tokenizer.encode('No', add_special_tokens=False)[0]
        self.yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        weave.init('outcome_similarity_detection')

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
        generated_scores = outputs.scores[0][0]

        no_score = generated_scores[self.no_token].item()
        yes_score = generated_scores[self.yes_token].item()
        scores = torch.tensor([no_score, yes_score])
        yes_probability = softmax(scores)[1].item()
        return yes_probability


class Text:
    def __init__(self, template):
        self.model = Generator()
        self.template = template

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        generated_text = self.model.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction


class Probability:
    def __init__(self, template, threshold=0.8):
        self.model = Generator()
        self.template = template
        self.threshold = threshold

    def predict(self, out1, out2):
        user_text = self.template.format(out1=out1, out2=out2)
        score = self.model.generate_probability(user_text)

        prediction = int(score >= self.threshold)
        return score, prediction
