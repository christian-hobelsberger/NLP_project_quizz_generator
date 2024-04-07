CL stands for Capitalized Letters, as this is the model which adds labels that are full words, i.e. 'Question: ' or 'Support:
'. 'large' 
refers to 'bart-large', the model used. In regards to the paper, this is the Single Finetuned Model fine-tuned on BART-large.

QA_CL_large_model.py in this folder is the code which resulted in the trained model utilized to generate the results from the
paper. It was run on the HPC Hábròk from the RUG, using a NVIDIA A100 GPU. It is heavily commented for reproducibility.

The outputs for the trained model which has been reported on can be found in the QA_CL_large_generator.out file.

In order to reproduce the model:
1. Move requirements.txt to a directory of your choice, and install the dependencies using: pip install -r
"DirectoryOfChoice/requirements.txt".
2. Open the QA_CL_large_model.py, and read comments for the global variables: change at least ALL the save paths.
3. Press run, it will first load and preprocess the SciQ dataset and show the filtering process in the output, and then start 
training. After training, it will generate some outputs on the unseen data split, however this is not part of the evaluation
of the paper, and no learning is involved in this step. 

In case a user wants to separately perform preprocessing and training, the main() function located at the bottom of the file
can be changed, and functionality for these steps can be found in the functions above it. The function 
'obtain_tokenized_filtered_splits' is directly responsible for the preprocessing of the SciQ dataset. The function
'training' is directly responsible for training. 

TestingQAModel.py in the main folder shows how to load and use the model to generate outputs, if you have the model and
specify the right path. 
