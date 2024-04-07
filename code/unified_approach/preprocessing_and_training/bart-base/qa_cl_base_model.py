# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:09:20 2024

@author: Rutger Lemein

One may note that this code and the large code file are nearly equivalent, except
for the selection of the model and parameters.
"""

# Libraries
import torch
import sys
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Save paths for the model
## Currently these are specific to the UG Habrok HPC, so change these if you want to run the code.
save_path_model_train = os.path.join(os.environ.get('TMPDIR'), 'results_CL_base', 'model_CL_base_train')
save_path_model_test = os.path.join(os.environ.get('TMPDIR'), 'results_CL_base', 'model_CL_base_test')

# Load pre-trained model and tokenizer
model_name = "facebook/bart-base"  # You can change the model name as needed
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Configurations
## Support as input, or support and answer as input? True is both, False is only support.
## The model discussed in the paper has this set to False.
support_and_answer = False

## Through testing, the best methodology is to provide BART with tags like 'Q:' for
## the question, 'A:' for the answer, 'D1:' for the distractor, etc. 
## This boolean turns the tags into full words, i.e. 'Question:', 'Answer:', 'Distractor1:'
## The model discussed in the paper has this set to True.
full_words = True

## Epochs, batch size, learning rate
num_epochs_train = 40
num_epochs_test = 2
batch_size = 8 
learning_rate = 5e-5

## Length of the input and output; the size to which they are padded or truncated.
length = 600



# Tokenizer functionality
def tokenize_func(examples, length=length):
    inputs = tokenizer(examples['merged_column_input'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    labels = tokenizer(examples['merged_column_output'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    
    return {
        'input_ids': inputs['input_ids'],
        'labels': labels['input_ids'],  
    }



# A function to obtain the tokenized data splits ready for training. 
def obtain_tokenized_filtered_splits(dataset):
    for split in dataset.keys():
        print(f"Processing {split} dataset...")
        sys.stdout.flush()
        current_dataset = dataset[split]
        
        # Separator id's seem to not work well for me. Maybe I misunderstand how they work.
        # However, BART is contextual, so what if we add contextual labels instead in the form of actual words and contextual symbols? WORKS
        if support_and_answer == False and full_words == False:
            support_labels =  ["S: "]*len(current_dataset["correct_answer"])
            question_labels = ["Q: "]*len(current_dataset["correct_answer"])
            answer_labels = ["A: "]*len(current_dataset["correct_answer"])
            distractor1_labels = ["D1: "]*len(current_dataset["correct_answer"])
            distractor2_labels = ["D2: "]*len(current_dataset["correct_answer"])
            distractor3_labels = ["D3: "]*len(current_dataset["correct_answer"]) 
            
            merged_column_input = [' '.join(row) for row in zip(support_labels, current_dataset["support"])]
            merged_column_output = [' '.join(row) for row in zip(question_labels, current_dataset["question"], answer_labels, current_dataset["correct_answer"], distractor1_labels, current_dataset["distractor1"], 
                                                          distractor2_labels, current_dataset["distractor2"], distractor3_labels, current_dataset["distractor3"])]
            
            # Add the merged columns to the dataset
            current_dataset = current_dataset.add_column('merged_column_input', merged_column_input)
            current_dataset = current_dataset.add_column('merged_column_output', merged_column_output)     
        
        
        # Another test is to include the answer in the input instead of the output. After all, if it's already working this well with only
        # the support as input, then how well would it work if we allow ourselves to include this as well?
        if support_and_answer == True and full_words == False:
            support_labels =  ["S: "]*len(current_dataset["correct_answer"])
            question_labels = ["Q: "]*len(current_dataset["correct_answer"])
            answer_labels = ["A: "]*len(current_dataset["correct_answer"])
            distractor1_labels = ["D1: "]*len(current_dataset["correct_answer"])
            distractor2_labels = ["D2: "]*len(current_dataset["correct_answer"])
            distractor3_labels = ["D3: "]*len(current_dataset["correct_answer"])      
            
            merged_column_input = [' '.join(row) for row in zip(support_labels, current_dataset["support"], answer_labels, current_dataset["correct_answer"])]
            merged_column_output = [' '.join(row) for row in zip(question_labels, current_dataset["question"], distractor1_labels, current_dataset["distractor1"], 
                                                          distractor2_labels, current_dataset["distractor2"], distractor3_labels, current_dataset["distractor3"])]
            
            # Add the merged columns to the dataset
            current_dataset = current_dataset.add_column('merged_column_input', merged_column_input)
            current_dataset = current_dataset.add_column('merged_column_output', merged_column_output)
           
            
        # I think maybe using the full word can help BART understand which parts of the input and output are which even better
        elif support_and_answer == True and full_words == True:
            support_labels =  ["Support: "]*len(current_dataset["correct_answer"])
            question_labels = ["Question: "]*len(current_dataset["correct_answer"])
            answer_labels = ["Answer: "]*len(current_dataset["correct_answer"])
            distractor1_labels = ["Distractor1: "]*len(current_dataset["correct_answer"])
            distractor2_labels = ["Distractor2: "]*len(current_dataset["correct_answer"])
            distractor3_labels = ["Distractor3: "]*len(current_dataset["correct_answer"])  
            
            merged_column_input = [' '.join(row) for row in zip(support_labels, current_dataset["support"], answer_labels, current_dataset["correct_answer"])]
            merged_column_output = [' '.join(row) for row in zip(question_labels, current_dataset["question"], distractor1_labels, current_dataset["distractor1"], 
                                                          distractor2_labels, current_dataset["distractor2"], distractor3_labels, current_dataset["distractor3"])]
            
            # Add the merged columns to the dataset
            current_dataset = current_dataset.add_column('merged_column_input', merged_column_input)
            current_dataset = current_dataset.add_column('merged_column_output', merged_column_output)
            
            
        # Lastly, also for the only support as input case
        elif support_and_answer == False and full_words == True:
            support_labels =  ["Support: "]*len(current_dataset["correct_answer"])
            question_labels = ["Question: "]*len(current_dataset["correct_answer"])
            answer_labels = ["Answer: "]*len(current_dataset["correct_answer"])
            distractor1_labels = ["Distractor1: "]*len(current_dataset["correct_answer"])
            distractor2_labels = ["Distractor2: "]*len(current_dataset["correct_answer"])
            distractor3_labels = ["Distractor3: "]*len(current_dataset["correct_answer"])   
            
            merged_column_input = [' '.join(row) for row in zip(support_labels, current_dataset["support"])]
            merged_column_output = [' '.join(row) for row in zip(question_labels, current_dataset["question"], answer_labels, current_dataset["correct_answer"], distractor1_labels, current_dataset["distractor1"], 
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



# Pretty standard training function
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



# Actually this function evaluates, not validates, that's where the mix-up mentioned in the paper between
# the data splits happened. It's not our final evaluation however; that is mostly qualitative or in the distractor script.
# This is just for some insight on performances on unseen data as the model was being built and trained.
def validate(model, eval_dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    batchcounter = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batchcounter += 1
        
        # Print validation outputs and reference outputs every 50 batches
        if batchcounter%50 == 0:
            print(f"Validation, batch: {batchcounter}")   
            # Generating outputs
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

    # Applying all the filters results in 10263 datapoints in train, 864 in test, 867 in val
    small_train_dataset = train_dataset.shuffle(seed=42).select(range(10256))
    small_test_dataset = test_dataset.shuffle(seed=42).select(range(864))
    small_eval_dataset = val_dataset.shuffle(seed=42).select(range(864))
    
    # Dataloaders required for the model
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(small_test_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    
    # Training
    train_model = training(train_dataloader, num_epochs_train, save_path_model_train, model)
    
    # Fine-tuning the trained model on testing data
    print("FINISHED TRAINING. NOW FINE-TUNING ON TEST DATA.\n")
    test_model = training(test_dataloader, num_epochs_test, save_path_model_test, train_model)

    # Last evaluation, no training, no epochs
    validate(test_model, eval_dataloader)

    
if __name__ == '__main__':
  main()