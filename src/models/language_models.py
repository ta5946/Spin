import torch
import weave
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity


# Pretrained language models
class SciBert:
    def __init__(self, threshold=0.8):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', device_map='cuda')
        self.threshold = threshold

    def get_vector(self, sentence):
        inputs = self.tokenizer.encode(sentence)
        inputs = torch.tensor(inputs).reshape(1, -1).to('cuda')

        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.last_hidden_state[:, 0, :].to('cpu')

    def predict(self, out1, out2):
        vector1 = self.get_vector(out1)
        vector2 = self.get_vector(out2)

        score = cosine_similarity(vector1, vector2)[0][0]
        prediction = int(score >= self.threshold)
        return score, prediction


class OlmoInstruct:
    def __init__(self, template):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-7B-Instruct-hf')
        self.model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-7B-Instruct-hf', torch_dtype=torch.bfloat16, device_map='cuda')
        self.template = template
        weave.init('outcome_similarity_detection')

    @weave.op
    def generate_text(self, user_text, n_tokens=10):
        chat = [
            {'role': 'user', 'content': user_text}
        ]
        input_tokens = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt').to('cuda')
        output_tokens = self.model.generate(input_tokens, max_new_tokens=n_tokens)[0]
        generated_tokens = output_tokens[input_tokens.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

    def predict(self, out1, out2):
        user_text = self.template.format(sentence1=out1, sentence2=out2)
        generated_text = self.generate_text(user_text)

        prediction = int('yes' in generated_text.lower())
        return prediction, prediction
