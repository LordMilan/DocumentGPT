#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


if __name__=="__main__":
        
        # Path to store faiss vector database
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        
        # loads the pdf 
        loader=PyPDFLoader("./documents/Milan_Mahat.pdf")
        docs=loader.load()
        
        #The text_splitter configured in this way will divide text into manageable chunks of 1000 characters each, with a 200-character overlap between adjacent chunks. This can be useful for processing large texts efficiently, especially in scenarios where the entire text cannot be processed at once due to memory constraints or computational limitations.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Define the path to the pre-trained model you want to use
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        model_kwargs = {'device':'cuda'}
        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )
        try:
            #Vector code of document is created and stored
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(DB_FAISS_PATH)
            print("Faiss index created ")
            
        except Exception as e:
            print("Fiass store failed \n",e)