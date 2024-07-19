import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)


def qa(prompt):
    context = "CWS CPU Usage increased\nEntitlement heap increased overtime\nBroker memory leak\n5xx error has increased\nPage service CPU usage increased overtime"
    system_msg = {
        "role": "system",
        "content": "Answer user's question using alerts given in the context. In the context are currently active alerts which describe the current status of the system. Answer with a formated list of active alerts, and with a short description for each active alert."
    }
    usr_msg = {
        "role": "user",
        "content": "Context:\n" + context + "\nQuestion: " + prompt
    }
    messages = [
        system_msg,
        usr_msg
    ]

    input_ids = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    terminators = [
        TOKENIZER.eos_token_id,
        TOKENIZER.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = MODEL.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.5,
        eos_token_id=terminators,
        do_sample=True,
        top_p=0.9
    )

    print("Answer:\n",TOKENIZER.decode(outputs[0][input_ids.shape[-1]:]))


while True:
    prompt = input("Question")
    print("Question: ", prompt)
    if prompt == 'exit':
        break
    qa(prompt)
