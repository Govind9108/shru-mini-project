from operator import itemgetter


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
# from key import *
from langchain_community.document_loaders import PyPDFLoader

import getpass
import os

 



class DocGptLangResponse:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def getDocSummary(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()
        
        vectorstore = FAISS.from_documents(pages, embedding=OpenAIEmbeddings())
        
        retriever = vectorstore.as_retriever()
        
        template = """You will be provided document context 
                and your task is to give the topic for it in single line also give the summary in nine to ten line.:
        {context}


        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI()
        
        chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )

        answer = chain.invoke("please give the topic for it in single line also give the summary in nine to ten line.")
        
        return answer
    
    def doc_context(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()
        
        page_contents = []

        page_contents = ""

        for page in pages:
            page_contents += page.page_content + "\n"  # Add page content with a newline separator
            
        # vectorstore = FAISS.from_documents(pages, embedding=OpenAIEmbeddings()) 
        # retriever = vectorstore.as_retriever()
        # context = retriever.invoke(pages)
        return page_contents