import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForCausalLM

MAX_INPUT_SIZE = 150
MAX_TARGET_SIZE = 1000

DATASET = load_dataset(
    'csv',
    data_files={'train': 'datasets/train.csv', 'test': 'datasets/test.csv'}
)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = torch.device("cuda:0")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)


class T5QADataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.inputs = [q for q in data['input']]
        self.outputs = [a for a in data['output']]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            f"answer a question: {self.inputs[idx]}",
            return_tensors='pt',
            max_length=MAX_INPUT_SIZE,
            padding='max_length',
            truncation=True
        ).to(DEVICE)

        target_encoding = self.tokenizer(
            self.outputs[idx],
            return_tensors='pt',
            max_length=MAX_TARGET_SIZE,
            padding='max_length',
            truncation=True,
        ).to(DEVICE)

        labels = target_encoding.input_ids.clone().squeeze()
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == TOKENIZER.pad_token_id] = -100

        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'labels': labels.squeeze(),
            'decoder_attention_mask': target_encoding.attention_mask.squeeze()
        }


def train():
    train_dataset = T5QADataset(TOKENIZER, DATASET['train'])
    test_dataset = T5QADataset(TOKENIZER, DATASET['test'])

    training_args = Seq2SeqTrainingArguments(
        output_dir='output',
        evaluation_strategy='epoch',
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.01
    )

    trainer = Seq2SeqTrainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=TOKENIZER
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(eval_results)

    # Save the model and tokenizer
    MODEL.save_pretrained('./fine_tuned_t5-v4')
    TOKENIZER.save_pretrained('./fine_tuned_t5-v4')


train()
