# compressionv2
NQ dataset download link:
https://ai.google.com/research/NaturalQuestions

Trivia dataset download link:
http://nlp.cs.washington.edu/triviaqa/

Llama-3.2-1B-Instruct link:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Llama-3.2-3B-Instruct link:
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

SentenceTransformer model link:
https://sbert.net/

Folder Structure:  
|root  
|---data_test  
|---model  
|---|---Llama-3.2-1B-Instruct  
|---|---Llama-3.2-3B-Instruct  
|---|---multi-qa-mpnet-base-cos-v1  
|---NQdataset  
|---trivia_dataset  
|---|---triviaqa-rc  
|---|---|---evidence  
|---|---|---qa  
|---compress_v2.py  
|---data_load.py  
|---dataset_process.py  
|---main.py  

download dataset, model first.
run data_load.py to load the data.
run main.py to see compressed result.
or run compress_v2.py use own documents to see the compressed result.
