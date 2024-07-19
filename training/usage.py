from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = './fine_tuned_t5-v4'
# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


# Function to generate an answer
def generate_answer(question):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=256,
        padding='max_length',
        truncation=True
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_length=256,
        num_beams=4,
        early_stopping=True)

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


# Example usage
questions = [
    "What is the difference between a process and a thread?",
    "What will happen in my system if CPU usage increases?",
    "My CPU usage has increased, what is the impact?",
    "How can I decrease the CPU usage?",
    "What is the capital of France?",
    "What is the capital city of France?",
    "What is France?",
    "What is Paris?",
    "Is Paris capital of France?",
    "Is Paris capital of France, yer or no?",
    "Name a movie",
    "What language does American speak?"
]
for question in questions:
    print("Q: " + question)
    answer = generate_answer("answer a question: " + question)
    print("A: " + answer)
    print("------------------------------------------------------------")
