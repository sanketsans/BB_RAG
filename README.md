# BB_RAG


This RAG(Retrieval Augmented Generation) model is developed using [Ollama](https://ollama.com/) models, speciafically I used [LLAMA2](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) from Ollama repositories. I tried other models - such as [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/), [LLAMA3](https://www.llama.com/) , but LLAMA2 seems to work fine in terms of speed and accuracy of the outputs. 

The RAG is based on the very popular web series [**Breaking Bad**](https://www.google.com/search?q=breaking+bad&rlz=1C5CHFA_enIT994IT994&gs_lcrp=EgZjaHJvbWUqBwgAEAAYjwIyBwgAEAAYjwIyDwgBEAAYQxixAxiABBiKBTIMCAIQABhDGIAEGIoFMgcIAxAAGIAEMgwIBBAAGEMYgAQYigUyBggFEEUYPDIGCAYQRRg8MgYIBxBFGDzSAQgxNzkwajBqN6gCALACAA&sourceid=chrome&ie=UTF-8&si=ACC90nxuVQO9WBG-fCGFSorfPFXlv7MTTYYmjOvUbIXaOaqU7EHSm79xGHccmGPJ_-zehWudU3D2dvlvaox2hXddDebXZXATN0HMrSTpNqahmyZ5vKSIyKc%3D&ictx=1&ved=2ahUKEwj7_YT8t-WIAxWwUGwGHa94DIsQyNoBKAB6BAgUEAA), which aired first in 2008 and spans over 5 seasons. The series has won multiple awards and been on top-watched list of series on Netflix. The datasets for all episodes is available on [Kaggle](https://www.kaggle.com/datasets/varpit94/breaking-bad-tv-show-all-seasons-episodes-data). 

Fortunately, I did not need to do any pre-processing on this dataset. I only used the summary section of the data and used that my text corpus for my model.

The text corpus are then used to extract their embeddings using the LLAMA2 model. 

I created a textual prompt format to spit out the output to user's queries : 

``
template = """ 
``
<br/>
``
Answer the questions based on the context provided. If you don't know the answer respond with "I don't know."
``
<br/>
``
Context : {context}
Question: {question}
"""
``

I use [FAISS - Facebook AI Similarity Search](https://ai.meta.com/tools/faiss/), to store the embeddings by creating a vector database and then use it store it elsewhere. 
One can also use the vector database to create external application. One good example is [here](https://python.langchain.com/docs/integrations/vectorstores/faiss/). 

I've been able to get some interesting outputs from the model.
eg. I inserted the prompt as ``query: Roast the resume ``
The output is receieved is : 
<br/>

Sure! Here's my attempt at roasting the resume of the character in Breaking Bad:\n\n"Oh boy, where do I even begin? Let\'s start with the highlights - or should I say, the lowlights. This guy has been involved in some serious shenanigans, from cooking meth to burying bodies. And let me tell you, his work experience is nothing to write home about either. He\'s worked as a high school chemistry teacher, a meth kingpin, and a suburban drug dealer. Talk about a resume gap!\n\nBut wait, it gets better. This guy has also been known to lie, cheat, and steal his way through life. He\'s got a serious problem with authority, and he\'s not afraid to break the law to get what he wants. And let me tell you, his interpersonal skills are lacking at best. He\'s got a tendency to manipulate and deceive those around him, all while maintaining a fake smile on his face.\n\nNow, I know what you\'re thinking - how could anyone with such a shady past get hired by any reputable company? Well, let me tell you, it\'s not easy. This guy has had to work hard to cover up his trail of destruction, and he\'s got some serious burn bridges behind him. But hey, at least he\'s consistent, right?\n\nIn conclusion, this resume is a real gem. It\'s got all the makings of a classic tale of woe - a troubled past, poor decision-making, and a general disregard for the law. So if you\'re looking for someone to join your team and cause some chaos, look no further! Just don\'t say I didn\'t warn you.

<br/>
I gave another prompt as ``query: Who is Elon Musk ``
The output is : 
<br/>
"I don't know who Elon Musk is. The name does not appear in any of the provided documents."

## To use the vector store : 

**STEP: I - Download the entire repo (this includes the folder)** 
<br/>
```
repo_id = "sanketsans/Breaking-Bad"
hf_hub_download(repo_id=repo_id, repo_type="dataset", revision="main", local_dir=local_dir, allow_pickle=True)
```
<br/>
<br/>

**STEP: II - Import FAISS and load the indexes**
<br/>
```
from langchain_community.vectorstores import FAISS

vector_store = FAISS.load_local(
    "faiss_index", embeds, allow_dangerous_deserialization=True
)
```
<br/>
<br/>
**STEP: III - Use as retriver / similarity search**
<br/>

```
retriever = vector_store.as_retriever()
```

## Additionally you can test the program using 
```
pip install -r requirements.txt
python test_bb_rag.py
```


