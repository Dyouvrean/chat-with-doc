from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import dotenv
import time
# creating a pdf reader object
dotenv.load_dotenv()
EMBEDDING_MODEL_NAME = "google/flan-t5-base"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
def split(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # the maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=50,  # the number of characters to overlap between chunks
        separators=MARKDOWN_SEPARATORS
    )
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    return text_splitter.split_text(doc)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
def to_vdb(doc):
    db= None
    for i, document in enumerate(doc):
        if db is None:
            # db = FAISS.from_texts(document, OpenAIEmbeddings(model="text-embedding-3-small"))
            db = FAISS.from_texts(document, embedding_model)
        else:
            db.add_texts(document)
        print(i)
        time.sleep(0)  # wait for 60 seconds before processing the next document

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return  retriever

def read_file(reader):
    text =""
    for page in reader.pages:
        text+=page.extract_text()
    return text
if __name__ == "__main__":


    file = "C:/Users/dry19/Desktop/Test .pdf"
    reader = PdfReader(file)
    doc = read_file(reader)
    splited = split(doc)
    print(len(splited))
    retriever = to_vdb(splited)
    query = input("Ask an question")
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)