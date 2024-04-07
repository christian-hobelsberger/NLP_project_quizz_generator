# Multiple Choice Question Generation by Fine-Tuning BART in Modular and Unified Approaches

## Overview
In education, creating quizzes and assessments is vital for evaluating student understanding. However, crafting high-quality multiple-choice questions can be challenging. Artificial intelligence (AI) technologies offer solutions to streamline question creation, saving time for educators and ensuring consistent assessment quality.

This project aims to design an AI-driven system to generate multiple-choice questions and distractors from textual input. The goal is to produce questions that accurately assess student understanding while providing challenging distractors.

## Approach
Our approach compares a single fine-tuned model that generates both questions and distractors against a component-wise approach. We aim to evaluate the effectiveness and efficiency of these two methodologies in producing high-quality questions and distractors.

 ## Used Data
For this project, we will be using the SciQ dataset, which can be found at [this link](https://huggingface.co/datasets/allenai/sciq). The SciQ dataset contains Question-Answer pairs along with three alternatives (also called distractors) and a context/support. The questions in this dataset originate from the fields of Physics, Chemistry, and Biology, among others.
