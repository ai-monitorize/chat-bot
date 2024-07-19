from langchain_core.runnables.base import RunnableSequence, RunnableMap
from langchain_core.prompts.prompt import PromptTemplate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFaceHub

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)

# Define the prompt template
prompt_template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

template = PromptTemplate(input_variables=["system_prompt", "user_prompt"], template=prompt_template)
llm = HuggingFaceHub(model_name=MODEL_ID, tokenizer=TOKENIZER, model=MODEL)

template_runnable = RunnableMap({"context": lambda x: x["context"], "question": lambda x: x["question"]})
qa_chain = RunnableSequence([template_runnable, template, llm])
# Function to answer questions
def qa(question):
    context = question["context"]
    response = qa_chain.run({"context": question, "question": question})
    print("Answer:\n", response)


# Main loop to ask questions
while True:
    prompt = input("Question: ")
    if prompt.lower() == 'exit':
        break
    qa(prompt)
