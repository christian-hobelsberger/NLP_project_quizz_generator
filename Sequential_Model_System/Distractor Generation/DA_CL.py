# -*- coding: utf-8 -*-
import torch
import sys
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Save paths for the model
save_path_model_train = os.path.join(os.environ.get('TMPDIR'), 'results_CL_large', 'model_CL_large_train')
save_path_model_test = os.path.join(os.environ.get('TMPDIR'), 'results_CL_large', 'model_CL_large_test')

# Load pre-trained model and tokenizer
model_name = "facebook/bart-base"  # You can change the model name as needed
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


## Through testing, the best methodology is to provide BART with tags like 'Q:' for
## the question, 'A:' for the answer, 'D1:' for the distractor, etc. 
## This boolean turns the tags into full words, i.e. 'Question:', 'Answer:', 'Distractor1:'
full_words = True

## Epochs, batch size, learning rate
num_epochs_train = 20
num_epochs_test = 2
batch_size = 8 
learning_rate = 5e-6

## Length of the input; so the size to which the predictors and predictee are padded
length = 600


def tokenize_func(examples, length=length):
    inputs = tokenizer(examples['merged_column_input'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    labels = tokenizer(examples['merged_column_output'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    
    return {
        'input_ids': inputs['input_ids'],
        # 'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids'],  # Labels for language modeling
    }




def detokenize_func(tokenizer, token_ids, label_ids):
    # Convert token IDs back to text
    tokenized_support = tokenizer.decode(token_ids, skip_special_tokens=False)
    tokenized_labels = tokenizer.decode(label_ids, skip_special_tokens=False)

    # Split the text using separator tokens
    parts = []
    for i in range(3):
        parts.append(tokenized_support.split(tokenizer.sep_token)[2 * i])
    
    for i in range(3):
        parts.append(tokenized_labels.split(tokenizer.sep_token)[2 * i])
    
    # Extract the question, correct answer, and distractors
    print(f"support: {parts[0]},\n question: {parts[1]},\n correct_answer: {parts[2]},\n distractor1: {parts[3]},\n distractor2: {parts[4]},\n distractor3: {parts[5]}")
    sys.stdout.flush()
    
    return {
        'support': parts[0],
        'question': parts[1],
        'correct_answer': parts[2],
        'distractor1': parts[3],
        'distractor2': parts[4],
        'distractor3': parts[5]
    }




def obtain_tokenized_filtered_splits(dataset):
    for split in dataset.keys():
        print(f"Processing {split} dataset...")
        sys.stdout.flush()
        current_dataset = dataset[split]
        
        support_labels =  ["Support: "]*len(current_dataset["correct_answer"])
        question_labels = ["Question: "]*len(current_dataset["correct_answer"])
        answer_labels = ["Answer: "]*len(current_dataset["correct_answer"])
        distractor1_labels = ["Distractor1: "]*len(current_dataset["correct_answer"])
        distractor2_labels = ["Distractor2: "]*len(current_dataset["correct_answer"])
        distractor3_labels = ["Distractor3: "]*len(current_dataset["correct_answer"])   
        
        merged_column_input = [' '.join(row) for row in zip(question_labels, current_dataset["question"], answer_labels, current_dataset["correct_answer"], support_labels, current_dataset["support"])]
        merged_column_output = [' '.join(row) for row in zip(distractor1_labels, current_dataset["distractor1"], 
                                                        distractor2_labels, current_dataset["distractor2"], distractor3_labels, current_dataset["distractor3"])]
        
        # Add the merged columns to the dataset
        current_dataset = current_dataset.add_column('merged_column_input', merged_column_input)
        current_dataset = current_dataset.add_column('merged_column_output', merged_column_output)
            
            
        # Filter the dataset to include only questions with supporting evidence for the correct answer (non-empty input)
        filtered_dataset = current_dataset.filter(lambda example: example['support'] is not None and example['support'] != "")
        # And include only questions with no superfluous information
        filtered_dataset = filtered_dataset.filter(lambda example: len(example['question']) < 171)
        # And remove any datapoints which contain questions that have a 'fill-in-the-blank' type answer
        filtered_dataset = filtered_dataset.filter(lambda example: '_______' not in example['question'] and '______' not in example['question'] and '_____' not in example['question']
                                                  and '____' not in example['question'] and '___' not in example['question'])
        
        # Run the tokenized_func on it to generate tokenized support (input) and merged_column (output)
        if split == 'train':
            tokenized_train = filtered_dataset.map(tokenize_func, batched=True)
        elif split == 'test':
            tokenized_test = filtered_dataset.map(tokenize_func, batched=True)
        else:
            tokenized_val = filtered_dataset.map(tokenize_func, batched=True)
        
        # Print the number of examples in the filtered dataset
        print(f"Number of examples with supporting evidence in {split} dataset:", len(filtered_dataset))
        sys.stdout.flush()
    
    return tokenized_train, tokenized_test, tokenized_val




def training(dataloader, num_epochs, save_path_model, model=model, learning_rate=learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(num_epochs):
        print(f"Training, epoch: {epoch}")
        sys.stdout.flush()
        # generated_questions = []
        # reference_questions = []
        total_loss = 0.0
        batchcounter = 0
        
        for batch in dataloader:
            batchcounter += 1
            if batchcounter%50 == 0:
                print(f"Training, batch: {batchcounter}")
                sys.stdout.flush()
            
            # Move tensors to the appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"]) # attention_mask=batch["attention_mask"],
            logits = outputs.logits

            # Compute the loss only on non-padding tokens
            active_loss = batch["labels"].view(-1) != tokenizer.pad_token_id
            active_logits = logits.view(-1, model.config.vocab_size)[active_loss]
            active_labels = batch["labels"].view(-1)[active_loss]
            
            # Calculate the cross-entropy loss
            loss = torch.nn.functional.cross_entropy(active_logits, active_labels)
            
            # loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            print(f"Loss: {loss}")
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            sys.stdout.flush()
        
        # To be able to observe how the model outputs between tests and between batches/epochs
        generated_output = model.generate(input_ids=batch["input_ids"], max_length=1024)
        for output in generated_output:
            detoken_output = tokenizer.decode(output, skip_special_tokens=False)
            print(f"Detokenized output: {detoken_output}")
        
        for reference in batch["labels"]:
            detoken_reference = tokenizer.decode(reference, skip_special_tokens=False)
            print(f"Reference output: {detoken_reference}")
            
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
        sys.stdout.flush()
            
    model.save_pretrained(save_path_model)
        
    return model




def validate(model, eval_dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    batchcounter = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batchcounter += 1
        if batchcounter%50 == 0:
            print(f"Validation, batch: {batchcounter}")        
            generated_output = model.generate(input_ids=batch["input_ids"], max_length=1024)
            for output in generated_output:
                detoken_output = tokenizer.decode(output, skip_special_tokens=False)
                print(f"Detokenized validation output: {detoken_output}")
    
            # Extracting reference questions
            for reference in batch["labels"]:
                detoken_reference = tokenizer.decode(reference, skip_special_tokens=False)
                print(f"Reference validation output: {detoken_reference}")
                



def main():
    # Load SciQ dataset
    dataset = load_dataset("allenai/sciq")
    
    # The function removes datapoints for which no support exists, and tokenizes the data to a constant processable
    # format: input_ids (tokenized support or tokenized support and answer) and label_ids (tokenized question and distractors (and answer))
    train_tokens, test_tokens, val_tokens = obtain_tokenized_filtered_splits(dataset)
    
    train_dataset = train_tokens.remove_columns(["question", "distractor3", "distractor1", "distractor2",
                                                 "correct_answer", "support", "merged_column_input", "merged_column_output"])
    test_dataset = test_tokens.remove_columns(["question", "distractor3", "distractor1", "distractor2",
                                                 "correct_answer", "support", "merged_column_input", "merged_column_output"])
    val_dataset = val_tokens.remove_columns(["question", "distractor3", "distractor1", "distractor2",
                                                 "correct_answer", "support", "merged_column_input", "merged_column_output"])
    
    # Make sure that we have our datasets in the right format
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")
    val_dataset.set_format("torch")
      
    # This is the Huggingface tutorial. Honestly I end up using all the data anyways
    # so I guess that this is not necessary, but whatever.
    # Applying all the filters results in 10263 datapoints in train, 864 in test, 867 in val
    small_train_dataset = train_dataset.shuffle(seed=42).select(range(10256))
    small_test_dataset = test_dataset.shuffle(seed=42).select(range(864))
    small_eval_dataset = val_dataset.shuffle(seed=42).select(range(864))
    
    # Dataloaders required for the model
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(small_test_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    
    # Training, 20 epochs
    train_model = training(train_dataloader, num_epochs_train, save_path_model_train, model)
    
    # Fine-tuning the trained model on testing data, 1 epochs
    print("FINISHED TRAINING. NOW FINE-TUNING ON TEST DATA.\n")
    test_model = training(test_dataloader, num_epochs_test, save_path_model_test, train_model)

    # Last validation, no training, no epochs
    validate(test_model, eval_dataloader)

    
if __name__ == '__main__':
  main()