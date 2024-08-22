# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# from app.chat.chat import Chat
#
#
# # llm = Llm("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
#
#
# class Llm:
#     def __init__(self, model_id: str, gguf_file=None) -> None:
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             # gguf_file=gguf_file,
#             device_map="cuda" if torch.cuda.is_available() else "cpu",
#         )
#         self.streamer = TextStreamer(self.tokenizer)
#
#     def generate_text(self, chat: Chat) -> str:
#         input_ids = self.tokenizer.apply_chat_template(
#             chat.messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.device)
#
#         terminators = [
#             self.tokenizer.eos_token_id,
#             self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]
#
#         for output in self.model.generate(
#                 input_ids,
#                 temperature=0.1,
#                 eos_token_id=terminators,
#                 streamer=self.streamer,
#                 # return_dict_in_generate=True,
#                 # output_scores=True,
#                 max_new_tokens=1024,
#                 do_sample=True,
#                 top_p=0.9,
#         ):
#             if output.dim() == 1:
#                 # Handle the case where output is a 1D tensor
#                 yield self.tokenizer.decode(output[input_ids.shape[-1]:])
#             elif output.dim() == 2:
#                 # Handle the case where output is a 2D tensor
#                 yield self.tokenizer.decode(output[0][input_ids.shape[-1]:])
#             # yield decoded_output
