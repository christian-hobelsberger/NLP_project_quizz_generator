WEB refers to the wolrd wide web, as this is the model that transforms extracted texts from URLs into supports.
In regards to the paper, this is the Web Extractor Model fine-tuned on BART-large.

QA_WEB_large_model.py in this folder is the code which resulted in the trained model utilized to generate the results from the paper. It was run on the HPC Hábròk from the RUG, using a NVIDIA A100 GPU.

Joint Pipeline Final.py in the main folder shows how to load and use the model to generate outputs, if you have the model and specify the right path. 
For this, you also need to have one of the MCQ generation models.

