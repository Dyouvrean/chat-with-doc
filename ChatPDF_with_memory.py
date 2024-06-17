from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import sys
import os
import tempfile

# Model set up
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
repo_id = "google/flan-t5-base"
llm = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_new_tokens": 100})


def load_doc(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    return docs


# Split the documents into smaller chunks
def split_file(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts


def get_chain(texts):
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity")
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           chain_type_kwargs={
                                               "prompt": prompt,
                                               "memory": memory
                                           })
    return qa_chain,memory


if __name__ == "__main__":
    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
    if uploaded_files:
        docs = load_doc(uploaded_files)
        texts = split_file(docs)
        qa_chain,memory = get_chain(texts)

        chat_history = []
        query = st.text_input("Prompt")
        if query.lower() in ["exit", "quit", "q"]:
            print('End the conversation')

        result = qa_chain.run(query)
        # custom_rag_prompt = PromptTemplate.from_template(template)
        # qa_chain = (
        #         {"context": retriever, "question": RunnablePassthrough()}
        #         | custom_rag_prompt
        #         | llm
        #         | StrOutputParser()
        # )
        # result = qa_chain.invoke(query)
        # print(custom_rag_prompt.format(question = query))
        # print(result)

        st.write('Answer: ' + result + '\n')
        st.write(memory.load_memory_variables({})['history'] + '\n')
        print(memory.load_memory_variables({}))
        chat_history.append((query, result))
        print(chat_history)