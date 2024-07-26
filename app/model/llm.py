import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


class ChatMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class Chat:
    def __init__(self, message: ChatMessage) -> None:
        self.chat = list()
        self.chat.append(message)

    def add_message(self, message: ChatMessage) -> None:
        self.chat[1] = message


class Llm:
    def __init__(self, model_id: str, torch_dtype: torch.dtype) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.streamer = TextStreamer(self.tokenizer)

    def generate_text(self, chat: Chat) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            chat.chat,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for output in self.model.generate(
                input_ids,
                temperature=0.1,
                eos_token_id=terminators,
                streamer=self.streamer,
                return_dict_in_generate=True,
                output_scores=True
        ):
            decoded_output = self.tokenizer.decode(output[0][input_ids.shape[-1]:])
            yield decoded_output
