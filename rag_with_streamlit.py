#import Essential dependencies
import os
import streamlit as sl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#function to load the vectordatabase
def load_knowledgeBase():
        
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
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        os.environ['OPENAI_API_BASE']  = "http://localhost:8080"
        llm = ChatOpenAI(model="stablelm-1.6", api_key="xxxx" )
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """ You need to answer the question using information from the context. 
        Context and question of the user is given below: 
        context: {context}
        question: {question}
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__=='__main__':
    knowledgeBase=load_knowledgeBase()
    llm=load_llm()

    prompt=load_prompt()   
    print("Faiss index loaded ")

    sl.header("Welcome to DocumentGPT 📄.")
    sl.write(" You can ask me your questions related to document")


    query=sl.text_input('What do you wanna know?')

    if(query):
        similar_documents=knowledgeBase.similarity_search(query)
        
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

        similar_embeddings=FAISS.from_documents(similar_documents, embeddings)
        print("Similar_embeddings is loaded")

        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        print("Retriever has run Successfully")
        # print(retriever | format_docs)
        print("rag_chain is running")
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        response=rag_chain.invoke(query)

        sl.write(response)

