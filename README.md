# NLP Project Group 7: Science Quiz Generation

## Overview

This project is largely the result of following the *Ideas for research directions* at the [IK-NLP 2024](https://sites.google.com/rug.nl/ik-nlp-2024/home) Project Description [Science Quiz Generation](https://sites.google.com/rug.nl/ik-nlp-2024/projects-description/science-quiz-generation).

The goal was to fine-tune existing language models for the purpose of automated multiple-choice quiz generation based on text input.


## Models
We fine-tuned various machine learning models. In pre-alpha, flan-t5 was trained for disparate question/answer and distractor generation in the _modular_ approach. In alpha, BART-base were used for similar modular training and also for the *unified* task, while BART-large was fine-tuned only on the unified task. To also tackle the challenge, the latter model was encapsulated for extracting input texts from webpages.

This repository contains the code used for preprocessing, training, and evaluating these models hosted on [Huggingface](https://huggingface.co/models):

**Pre-alpha**:
- [question-generation](https://huggingface.co/rizkiduwinanto/question-generation)
- [distractor-generation](https://huggingface.co/rizkiduwinanto/distractor-generation)

**Alpha**:
- [final-bart-question-generation](https://huggingface.co/rizkiduwinanto/final-bart-question-generation)
- [final-bart-distractor-generation](https://huggingface.co/rizkiduwinanto/final-bart-distractor-generation)
- [CL_base](https://huggingface.co/b-b-brouwer/CL_base)
- [CL_large](https://huggingface.co/b-b-brouwer/CL_large)

**Beta/challenge**:
- [WEB_large](https://huggingface.co/rizkiduwinanto/WEB_large)

## Data
These models were trained on [SciQ](https://huggingface.co/datasets/allenai/sciq). This dataset contains Question-Answer pairs along with three alternatives (also called distractors) and a context/support on which the question is based. The questions are crowedsourced science exam questions.

## Requirements
Run `pip install -r requirements.txt` for the file `requirements.txt` included on the repo.

## Preprocessing and Training
*File paths relative to* `code`

The code for preprocessing of SciQ is available in `evaluation_both_approaches/pipeline_output_detokenized.ipynb` under the heading Helper class EvaluationConfigs > Function definitions. These functions depend on a tokenizer and source/target token token length arguments.


Equivalent functionality is (unfortunately) for the BART-based **Alpha** and **Beta/challenge** models repeated in the code used for training:

- final-bart-question-generation: `modular_approach/fine_tuning_bart_base/distractors/da_cl.py`
- final-bart-distractor-generation: `modular_approach/fine_tuning_bart_base/question_answer/qa_cl.py`
- CL_base: `unified_approach/preprocessing_and_training/bart-base/qa_cl_base_model.py`
- CL_large: `unified_approach/preprocessing_and_training/bart-large/qa_cl_large_model.py`


## Model Inference and Evaluation
*File paths relative to* `code`

Model inference procedures are demonstrated in the Jupyter notebooks in `evaluation_both_approaches` and `unified_approach/challenge` (using the Huggingface API to load the models) and in `.py` files in `unified_approach/evaluation` (assuming local models). The Jupyter notebook `pipeline_output_detokenized.ipynb` is geared towards generating and saving pandas DataFrames of detokenized outputs, while `distractors_metrics_quantitative.ipynb` demonstrates the procedure for quantitative evaluation using the BLUE metric, among others. The detokenized outputs are human-readible and can be evaluated according to the criteria proposed by [Tarrant et al. (2006)](https://hub.hku.hk/bitstream/10722/54324/1/134913.pdf?accept=1).