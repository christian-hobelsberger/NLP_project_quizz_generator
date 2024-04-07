CL stands for Capitalized Letters, as this is the model which adds labels that are full words, i.e. 'Question: ' or 'Support: '. 'large' 
refers to 'bart-large', the model used. In regards to the paper, this is the Single Finetuned Model fine-tuned on BART-large.

QA_CL_large_model.py in this folder is the code which resulted in the trained model utilized to generate the results from the paper. It was run on the HPC Hábròk from the RUG, using a NVIDIA A100 GPU.

TestingQAModel.py in the main folder shows how to load and use the model to generate outputs, if you have the model and specify the right path. 

