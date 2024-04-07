# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:59:15 2024

@author: Rutger Lemein

"""

import requests
from bs4 import BeautifulSoup
from googlesearch import search
import torch
import sys
import os
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_dataset
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
import pickle
import re
from langdetect import detect

# Tokenization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Model Configuration
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Save paths for the model
## For my pc
# save_path_model = "C:/Users/DLeme/Downloads/"
# save_path_model_train = save_path_model
# save_path_model_test = save_path_model
# save_path_data = save_path_model
# save_path_data_raw = os.path.join("C:\\", "Users", "DLeme", "Downloads", "raw_data")
# load_path_data = os.path.join("C:\\", "Users", "DLeme", "Downloads")

## For Habrok
save_path_model_train = os.path.join(os.environ.get('TMPDIR'), 'results_WEB_large', 'model_WEB_large_train')
save_path_model_test = os.path.join(os.environ.get('TMPDIR'), 'results_WEB_large', 'model_WEB_large_test')
save_path_data = os.path.join(os.environ.get('TMPDIR'), 'results_WEB_large', 'google_data')
save_path_data_raw = os.path.join(os.environ.get('TMPDIR'), 'results_WEB_large', 'google_data', 'raw_data')
load_path_data = os.path.join(os.environ.get('TMPDIR'), 'results_WEB_large', 'google_data')


## Epochs, batch size, learning rate
num_epochs_train = 100
num_epochs_test = 2
batch_size = 8 
learning_rate = 5e-6

## Length of the input; so the size to which the predictors and predictee are padded
## After filtering, inputs go up to about a maximum of 50000 characters.
## 4.7 average characters per word, so 50000/4.7 = 10638 length
## That's a little too much, and I'd rather truncate when text gets very long.
## After all it's just randomly selecting any text that fits the criteria of containing the topic.
## So not using all of it is not a shame at all, especially if there's a lot. Hopefully a lesser amount
## will contain enough information so that it can be transformed into the support of the SciQ set, hopefully.
## BART has a limit of 1024, so that's what we will use.
length = 1024

### Not doing this right now because I upped the minimum wiki length and so I just don't know how much will be cut; we'll use all the remaining data and see.
## However, a lot of data points will be dropped as they are too short in size (see below). So maybe it will be a little better.
## In any case, it seems risky to leave the dataloaders as big as they are, given that with 100 of each split, 30% of the data points
## were eliminated in the filtering process. So let's take a good 35% elimination as a safety measure (we do not want to have to run the model again), 
## then we have 11679 * 0.65 = 7591 in train, and 650 in test and validation. This should also reduce training time by approx 35%, so 15 hours now (YAY).
## 45 hour script it is. 
# data_size_train = 7584
# data_size_test = 648
# data_size_validate = 648

## This one controls the minimum length of text to be accepted as input. This is important,
## because some questions have silly answers that cannot serve as a topic because either the answer
## is too general, like 'brain', or the answer is too abstract, like '40 percent', and results in a
## Wikipedia reference page with a very low number of characters (30-100 usually). These pages must be avoided at all costs.
## Very short texts in general are likely to not contain enough information to train the model with. 
## 600 characters is about 128 words minimum (so essentially BART will have a minimum input length of ~128 (words are not exactly word tokens but aight)).
## Note that this variable is also used to filter out any empty inputs, however those empty inputs will have some
## label characters! So be sure to keep this above 300 at least.

minimum_wiki_length = 600



def tokenize_func(examples, length=length):
    inputs = tokenizer(examples['merged_column_input'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    labels = tokenizer(examples['merged_column_output'], return_tensors="pt", max_length=length, truncation=True, padding='max_length')
    
    return {
        'input_ids': inputs['input_ids'],
        # 'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids'],  # Labels for language modeling
    }




def topic_contains_number(topic):
    # Regular expression to match any digit in the topic
    regex = re.compile(r'\d')
    return bool(regex.search(topic))

def is_written_out_number(topic):
    written_out_numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'] 
    return topic.lower() in written_out_numbers

def obtain_tokenized_data(dataset, minimum_wiki_length=minimum_wiki_length, save_path_data=save_path_data, use_raw_data=False, save_raw_data=False, relevant_texts_train=False, relevant_texts_test=False, relevant_texts_validation=False):
    for split in dataset.keys():
        print(f"Processing {split} dataset...")
        current_dataset = dataset[split]
        
        # First generate the texts
        topics = current_dataset["correct_answer"]
        if relevant_texts_train == True and split == 'train':
            # Load relevant_texts from the pickle file
            load_path_data_train = os.path.join(load_path_data, f"relevant_texts_{split}.pkl")
            with open(load_path_data_train, 'rb') as f:
                relevant_texts = pickle.load(f)
        elif relevant_texts_test == True and split == 'test':
            # Load relevant_texts from the pickle file
            load_path_data_test = os.path.join(load_path_data, f"relevant_texts_{split}.pkl")
            with open(load_path_data_test, 'rb') as f:
                relevant_texts = pickle.load(f)
        elif relevant_texts_validation == True and split == 'validation':
            # Load relevant_texts from the pickle file
            load_path_data_val = os.path.join(load_path_data, f"relevant_texts_{split}.pkl")
            with open(load_path_data_val, 'rb') as f:
                relevant_texts = pickle.load(f)
                
        else:
            relevant_texts = []
            for topic in topics:
                if not topic_contains_number(topic) and not is_written_out_number(topic):
                    relevant_text = obtain_relevant_text(topic, use_raw_data, save_raw_data)
                else: 
                    relevant_text = None
                relevant_texts.append(relevant_text)
            if len(relevant_texts) == len(topics):
                print("Good to go.")
                sys.stdout.flush()            
                # Save relevant_texts to a file as Google does not like all these requests one bit
                save_path_data_new = os.path.join(save_path_data, f"relevant_texts_{split}.pkl")
                with open(save_path_data_new, 'wb') as f:
                    pickle.dump(relevant_texts, f)
        
        text_labels = ["Text: "]*len(current_dataset["correct_answer"])
        support_labels =  ["Support: "]*len(current_dataset["correct_answer"])
        question_labels = ["Question: "]*len(current_dataset["correct_answer"])
        answer_labels = ["Answer: "]*len(current_dataset["correct_answer"])
        
        # Now these texts are inputs, and supports are outputs
        merged_column_input = [' '.join(row) if all(part is not None for part in row) else '' for row in zip(answer_labels, current_dataset["correct_answer"], text_labels, relevant_texts)]
        merged_column_output = [' '.join(row) for row in zip(question_labels, current_dataset["question"], support_labels, current_dataset["support"])]
        
        # Add the merged columns to the dataset
        current_dataset = current_dataset.add_column('merged_column_input', merged_column_input)
        current_dataset = current_dataset.add_column('merged_column_output', merged_column_output)
        
        # Filter the dataset to include only questions with relevant text for the support
        filtered_dataset = current_dataset.filter(lambda example: example['merged_column_input'] != "" and example['merged_column_input'] is not None and len(example['merged_column_input']) > minimum_wiki_length)
        # Filter the dataset to include only questions with supporting evidence for the correct answer (non-empty input)
        filtered_dataset = filtered_dataset.filter(lambda example: example['support'] is not None and example['support'] != "")
        # And include only questions with no superfluous information
        filtered_dataset = filtered_dataset.filter(lambda example: len(example['question']) < 171)
        # And remove any datapoints which contain questions that have a 'fill-in-the-blank' type answer
        filtered_dataset = filtered_dataset.filter(lambda example: '_______' not in example['question'] and '______' not in example['question'] and '_____' not in example['question']
                                                  and '____' not in example['question'] and '___' not in example['question'])        
        
        # Run the tokenized_func on it to generate tokenized input and output
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
                sys.stdout.flush()
    
            # Extracting reference questions
            for reference in batch["labels"]:
                detoken_reference = tokenizer.decode(reference, skip_special_tokens=False)
                print(f"Reference validation output: {detoken_reference}")
                sys.stdout.flush()
                
                
 
                
def google_search_wikipedia(topic, max_retries=1, initial_delay=2000):
    # There is a search limit, which makes things hard. 
    # I think it's 300 requests per ~33.333 minutes.
    # But that's just a guess based on some tests, search_counter cancels at 299,
    # and the time it took to be able to request again is about 2000 seconds, but idk exactly.
    # If it works with the current setup I can process the 13679 data points in roughly 25.41 hours.
    query = f"{topic} Wikipedia"
    retries = 0
    while retries < max_retries:
        try:
            search_results = search(query, num_results=5)
            for url in search_results:
                if "wikipedia.org" in url:
                    return url
            print("No Wikipedia page found for the topic:", topic)
            sys.stdout.flush()
            return None
        except Exception as e:
            print("Error occurred during Google search:", str(e))
            # Increase delay exponentially with each retry
            delay = initial_delay * (2 ** retries)
            print(f"Retrying in {delay} seconds...")
            sys.stdout.flush()
            time.sleep(delay)
            retries += 1
            if retries > max_retries:
                print(f"Exceeded maximum number of retries ({max_retries}) to fetch data using exponential backoff.")
                sys.stdout.flush()
                return None




def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from all paragraph tags
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            return ' '.join(paragraphs)  # Concatenate paragraphs into a single text
        else:
            print("Failed to fetch URL:", url)
            sys.stdout.flush()            
            return None
    except Exception as e:
        print("Error occurred while fetching URL:", str(e))
        sys.stdout.flush()
        return None




def filter_paragraphs_by_topic(text, topic):
    relevant_paragraphs = []
    try:
        language_text = detect(text)
        if (language_text == 'en'):
            # Split the text into paragraphs based on newline characters
            paragraphs = text.split('\n')
            
            # Simple, simply keep the paragraph if it contains the topic
            for paragraph in paragraphs:
                if topic.lower() in paragraph.lower():
                    relevant_paragraphs.append(paragraph)
                    
            # However, simple doesn't always work, so
            if len(relevant_paragraphs) == 0:
                topics = topic.split(' ')
                for split_topic in topics:
                    for paragraph in paragraphs:
                        if split_topic.lower() in paragraph.lower() and paragraph not in relevant_paragraphs:
                            relevant_paragraphs.append(paragraph)
        else:
            print("The text fetched for this topic is sadly not in English, so we cannot use it.")
            sys.stdout.flush()
            return None
                    
        # If that doesn't work... need to think of something else
        # Additionally, certain paragraph filters basically return
        # the entire text, so it really isn't a perfect method.
        # Sometimes it'll return a text of length 4000, other times of
        # length 40000.
    except:
        print("No text was fetched for this topic, as no Wikipedia page was found, so we cannot filter it.")
        sys.stdout.flush()
        return None
        
                
    return ' '.join(relevant_paragraphs)




def obtain_relevant_text(topic, use_raw_data=False, save_raw_data=False):
    if use_raw_data == False:
        search_results = google_search_wikipedia(topic)
        url = search_results        
        text = extract_text_from_url(url)
    if use_raw_data == True:
        try:
            save_path_data_raw_new = os.path.join(save_path_data_raw, f"{topic}.txt")
            with open(save_path_data_raw_new, 'r') as file:
                try:
                    text = file.read()  
                except:
                    print("Weird character... sad I can't load those files I guess.")
                    text = None
        except:
            print("No file for this topic.")
            text = None

    if save_raw_data == True and text:    
        # Save the text to a file
        save_path_data_raw_new = os.path.join(save_path_data_raw, f"{topic}.txt")
        with open(save_path_data_raw_new, 'w', encoding='utf-8') as file:
            file.write(text)
        
        
    relevant_text = filter_paragraphs_by_topic(text, topic)
    # print(relevant_text)
    if relevant_text:
        print(f"Length original text: {len(text)}")
        print(f"Length relevant text: {len(relevant_text)}")
        sys.stdout.flush()
    
    
    return relevant_text




def main():
    # topic1 = "mesophilic organisms"
    # topic2 = "coriolis effect"
    # topic3 = "alpha decay"
    # topic4 = "meteorite"
    # topic5 = "radioactivity"
    # relevant_text = obtain_relevant_text(topic1)
    # relevant_text = obtain_relevant_text(topic2)
    # relevant_text = obtain_relevant_text(topic3)
    # relevant_text = obtain_relevant_text(topic4)
    # relevant_text = obtain_relevant_text(topic5)
    
    # So, the text maker works. Now, a model that takes these types of text
    # and transforms them into supports.
    # Load SciQ dataset
    dataset = load_dataset("allenai/sciq")
    
    # To run it on my own pc
    # dataset['train'] = dataset['train'].select(range(100))
    # dataset['test'] = dataset['test'].select(range(90))
    # dataset['validation'] = dataset['validation'].select(range(100))
    
    # First time? 
    # train_tokens, test_tokens, val_tokens = obtain_tokenized_data(dataset, use_raw_data=False, save_raw_data=True, relevant_texts_train=False, relevant_texts_test=False, relevant_texts_validation=False)
    
    # Already got the data?
    train_tokens, test_tokens, val_tokens = obtain_tokenized_data(dataset, use_raw_data=False, save_raw_data=False, relevant_texts_train=True, relevant_texts_test=True, relevant_texts_validation=True)
    
    # New filtering on old raw_data resulting in None values? .pkl files will be saved.
    # train_tokens, test_tokens, val_tokens = obtain_tokenized_data(dataset, use_raw_data=True, save_raw_data=False, relevant_texts_train=False, relevant_texts_test=False, relevant_texts_validation=False)


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
    
    # small_train_dataset = train_dataset.shuffle(seed=42).select(range(data_size_train))
    # small_test_dataset = test_dataset.shuffle(seed=42).select(range(data_size_test))
    # small_val_dataset = val_dataset.shuffle(seed=42).select(range(data_size_validate))
    
    # Dataloaders required for the model
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training, 20 epochs
    train_model = training(train_dataloader, num_epochs_train, save_path_model_train, model)
    
    # Fine-tuning the trained model on testing data, 1 epochs
    print("FINISHED TRAINING. NOW FINE-TUNING ON TEST DATA.\n")
    test_model = training(test_dataloader, num_epochs_test, save_path_model_test, train_model)

    # Last validation, no training, no epochs
    validate(test_model, eval_dataloader)
            
    
if __name__ == '__main__':
  main()