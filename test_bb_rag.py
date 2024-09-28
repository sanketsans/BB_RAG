
import argparse
import subprocess
import os
from huggingface_hub import snapshot_download


## arguments using argparse 
parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model", type=str, default="llama2", help="Model name")
args = parser.parse_args()

def check_and_download_ollama_model(model_name):
    """
    Checks if the model is already present in the repository.
    If the model is not found, it downloads the model.
    Args:
        model_name (str): The name of the model to check/download
    Returns:
        bool: True if the model is already present, False if the model needs to be downloaded
    """
    # Check if the model is already present in the repo
    try:
        # List the models in the repo
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        # Split the string into lines
        lines = result.stdout.split('\n')

        # Extract the names from each line (removing leading/trailing whitespace and the first line)
        models = [line.strip() for line in lines[1:] if line.strip()]
        status = False
        for model in models:
            if model_name in model:
                status = True
        print(models, status, model_name)
        
        # Check if the model is in the list
        if status:
            print(f"The model '{model_name}' is already present in the repository.")
            return True
        else:
            print(f"The model '{model_name}' is not found. Downloading...")
            # Download the model
            subprocess.run(['ollama', 'pull', model_name])
            print(f"The model '{model_name}' has been downloaded successfully.")
            return False
    except Exception as e:
        print(f"Model name is wrong: {e}")
        return False

status = check_and_download_ollama_model("llama2")
if not status:
    print('Try again with correct model name or Run the program again to load the model downloaded.')
else:

    from operator import itemgetter 
    from langchain_community.llms import Ollama 
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import PromptTemplate 

    model = Ollama(model=args.model) ## llama2
    embeds = OllamaEmbeddings() ## it differs for other models used. 
    parser = StrOutputParser()


    template = """
    Answer the questions based on the context provided. If you are not sure, respond with "I don't know."

    Context : {context}
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    print(prompt.format(context="Here is some context", question="Here is the question"))

    repo_id = "sanketsans/Breaking-Bad"
    local_dir = os.getcwd()
    snapshot_download(repo_id=repo_id,local_dir=local_dir,repo_type="dataset")

    vector_store =FAISS.load_local(
        "faiss_index", embeds, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()

    chain = (
        {"context": itemgetter("question") | retriever,
        "question": itemgetter("question")}
        | prompt | model | parser
    )
    while True : 
        question = input("Prompt anything BB: ")
        print(chain.invoke({"question": question}))

    # chain.invoke({"question": "Roast the resume"})


