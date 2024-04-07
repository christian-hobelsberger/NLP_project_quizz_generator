# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:38:04 2024

@author: Rutger Lemein

Playing around with my model.
"""

import requests
from bs4 import BeautifulSoup
import sys
from transformers import BartForConditionalGeneration, BartTokenizer
import re
import wikipediaapi

# User key for the Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MCQ Generation (r.j.a.lemein@student.rug.nl)', 'en')

# Define the path where the model is saved
save_path_QA_model_base = "C:/Users/Rutger Lemein/Downloads/Single_Fine_Tuned_System/Models/CL_base/model_CL_test"
save_path_QA_model_large = "C:/Users/Rutger Lemein/Downloads/Single_Fine_Tuned_System/Models/CL_large/model_CL_large_test"
save_path_WEB_model_base = "C:/Users/Rutger Lemein/Downloads/Single_Fine_Tuned_System/Models/WEB_base/model_WEB_test"
save_path_WEB_model_large = "C:/Users/Rutger Lemein/Downloads/Single_Fine_Tuned_System/Models/WEB_large/model_WEB_large_test"

# Load the model from the saved path
## Base models
model_QA_base = BartForConditionalGeneration.from_pretrained(save_path_QA_model_base)
model_WEB_base = BartForConditionalGeneration.from_pretrained(save_path_WEB_model_base)
model_name_base = "facebook/bart-base"  # You can change the model name as needed
tokenizer_base = BartTokenizer.from_pretrained(model_name_base)

## Large models
model_QA_large = BartForConditionalGeneration.from_pretrained(save_path_QA_model_large)
model_WEB_large = BartForConditionalGeneration.from_pretrained(save_path_WEB_model_large)
model_name_large = "facebook/bart-large"  # You can change the model name as needed
tokenizer_large = BartTokenizer.from_pretrained(model_name_large)
    

def generate_QA(input_text, bart_large=False):
    if bart_large == True:
        # Tokenize the input text
        input_ids = tokenizer_large(input_text, return_tensors="pt")["input_ids"]
        
        # Generate outputs
        outputs = model_QA_large.generate(input_ids=input_ids, max_length=1024)
    
        # Decode the generated outputs
        generated_text = tokenizer_large.decode(outputs[0], skip_special_tokens=True)
    
        # Print the generated text
        print(f"MCQ generation from generated support: {generated_text}\n")
        
        return generated_text
    else:
        # Tokenize the input text
        input_ids = tokenizer_base(input_text, return_tensors="pt")["input_ids"]
        
        # Generate outputs
        outputs = model_QA_base.generate(input_ids=input_ids, max_length=1024)
    
        # Decode the generated outputs
        generated_text = tokenizer_base.decode(outputs[0], skip_special_tokens=True)
    
        # Print the generated text
        print(f"MCQ generation from generated support: {generated_text}\n")
        
        return generated_text


def generate_support(url, topic, direct_wiki=False, bart_large=False, not_print=False):
    if bart_large == True:
        # Obtain input
        if direct_wiki==False:
            text = extract_text_from_url(url)
            filtered_text = filter_paragraphs_by_topic(text, topic)
        elif direct_wiki==True:
            display_title = url.split("/")[-1]
            if not_print == False:
                print(display_title)
            page = wiki_wiki.page(display_title)
            plain_text_content = page.text
            filtered_text = filter_paragraphs_by_topic(plain_text_content, topic)
        
        text_label = "Text: "
        answer_label = "Answer: "
        
        # Now these texts are inputs, and supports are outputs
        merged_column_input = f"{answer_label} {topic} {text_label} {filtered_text}"
    
        # Tokenize the filtered text. Note that large has the topic as an input as well!
        input_ids = tokenizer_large(merged_column_input, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
        
        # Generate outputs
        outputs = model_WEB_large.generate(input_ids=input_ids, max_length=1024)
    
        # Decode the generated outputs
        generated_text = tokenizer_large.decode(outputs[0], skip_special_tokens=True)
    
        # Print the generated text
        if not_print == False:
            print(f"Output of url to support generator: {generated_text}\n")
        
        # Extract the support
        pattern = r'Support:.*$'
        
        # Use regular expression to find the question
        matcher = re.search(pattern, generated_text)
        
        if matcher:
            support = matcher.group(0)
            support.strip()
            if not_print == False:
                print(f"Found support in output: {support}\n")  # Remove leading/trailing whitespace
            return support
        else:
            if not_print == False:
                print("No support found in the output string. The next model will be fed with all generated text.\n This might cause strange results.")
            return generated_text
    else:
        # Obtain input
        if direct_wiki==False:
            text = extract_text_from_url(url)
            filtered_text = filter_paragraphs_by_topic(text, topic)
        elif direct_wiki==True:
            display_title = url.split("/")[-1]
            if not_print == False:
                print(display_title)
            page = wiki_wiki.page(display_title)
            plain_text_content = page.text
            filtered_text = filter_paragraphs_by_topic(plain_text_content, topic)
        
        text_label = "Text: "
        answer_label = "Answer: "
        
        # Now these texts are inputs, and supports are outputs
        merged_column_input = f"{answer_label} {topic} {text_label} {filtered_text}"
    
        # Tokenize the filtered text. Note that large has the topic as an input as well! Base not for the old data. For the new data I'm training a base that only
        # produces Support, but really I should make them still produce both question and support.... oh well.
        input_ids = tokenizer_base(merged_column_input, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
        
        # Generate outputs
        outputs = model_WEB_base.generate(input_ids=input_ids, max_length=1024)
    
        # Decode the generated outputs
        generated_text = tokenizer_base.decode(outputs[0], skip_special_tokens=True)
    
        # Print the generated text
        if not_print == False:
            print(f"Output of url to support generator: {generated_text}\n")
        
        # Extract the support
        pattern = r'Support:.*$'
        
        # Use regular expression to find the question
        matcher = re.search(pattern, generated_text)
        
        if matcher:
            support = matcher.group(0)
            support.strip()
            if not_print == False:
                print(f"Found support in output: {support}\n")  # Remove leading/trailing whitespace
            return support
        else:
            if not_print == False:
                print("No support found in the output string. The next model will be fed with all generated text.\n This might cause strange results.")
            return generated_text
    


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
                    
    except:
        print("No text was fetched for this topic, as no page was found, so we cannot filter it.")
        sys.stdout.flush()
        return None
        
                
    return ' '.join(relevant_paragraphs)


def generate_QA_from_url(url, topic, direct_wiki=False, bart_large=True, not_print_support=True):
    generate_QA(generate_support(url, topic, direct_wiki=direct_wiki, bart_large=bart_large, not_print=not_print_support), bart_large=bart_large)
    

def main(): 
    # Some examples
    generate_QA_from_url("https://en.wikipedia.org/wiki/Energy", "energy", direct_wiki=True, not_print_support=False)
    generate_QA_from_url("https://en.wikipedia.org/wiki/Geology", "geology", direct_wiki=True, not_print_support=False)
    generate_QA_from_url("https://en.wikipedia.org/wiki/Ecology", "ecology", direct_wiki=True, not_print_support=False)    
    generate_QA_from_url("https://en.wikipedia.org/wiki/Ecology", "biodiversity", direct_wiki=True, not_print_support=False) 
    
    # Non Wikipedia example
    print("Non Wikipedia example:")
    generate_QA_from_url("https://science.nasa.gov/universe/black-holes", "black hole", not_print_support=False)

if __name__ == '__main__':
    main()