WEB refers to the world wide web, as this is the model that transforms extracted texts from URLs into supports.
In regards to the paper, this is the Web Extractor Model fine-tuned on BART-large.

QA_WEB_large_model.py in this folder is the code which resulted in the trained model utilized to generate the results from the paper. It was run on the HPC Hábròk from the RUG, using a NVIDIA A100 GPU. It is heavily commented for reproducibility.

The outputs for the trained model which has been reported on can be found in the QA_WEB_large_generator.out file.

In order to reproduce the model:
1. Move requirements.txt to a directory of your choice, and install the dependencies using: pip install -r "DirectoryOfChoice/requirements.txt".
2. Unpack the data hosted at https://drive.google.com/file/d/1DLiqLJVr8PIYxid7phJxxYiFir_oEnOB/view?usp=sharing 
3. Open the QA_WEB_large_model.py, and read comments for the global variables: change at least ALL the save paths.
4. Press run, it will first preprocess the data and show the filtering process in the output, and then start training. After training, it will generate some outputs on the unseen data split, however this is not part of the evaluation of the paper, and no learning is involved in this step. .pkl files will be saved for the processed .txt files, so that one can run it again much faster, see the parameters and the comments in the QA_WEB_large_model.py file.

Joint Pipeline Final.py in the main folder shows how to load and use the model to generate outputs, if you have the model and specify the right path. 
For this, you also need to have one of the MCQ generation models, and specify the right path.

In case a user wants to separately perform preprocessing and training, the main() function located at the bottom of the file can be changed, and functionality for these steps can be found in the functions above it. The function 'obtain_tokenized_data' is directly responsible for the preprocessing of the raw data or .pkl files. The function 'training' is directly responsible for training. 
