

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# loading PDF files and splitting the text
def load_and_split_pdf(file_path: str) -> list:
    """
    Load a PDF file and split its text into smaller chunks.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        List[Document]: A list of Document objects containing the split text.
        """

    # Load the PDF file
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)

    return split_docs


# Create a Chroma vector store
def store_chunks(chunks):
    """
    Create a Chroma vector store from the given chunks.
    Args:
        chunks (List[Document]): A list of Document objects containing the split text.
    Returns:
        Chroma: A Chroma vector store containing the embedded chunks.
    """
    # Create embeddings using HuggingFaceEmbeddings
    # You can choose a different model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create a Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"  # Directory to store the vector database
    )
    return vector_store



# Load the vector store from the directory
def load_qa_chain(db):
    """
    Load the QA chain using the vector store and the Mistral model.
    Args:
        db (Chroma): The Chroma vector store.
    Returns:
        RetrievalQA: A RetrievalQA chain that uses the Mistral model for question answering.
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    # Set the chain type to "stuff" for simple question answering
    # You can also use "map_reduce" or "refine" for more complex scenarios

    return qa


def extract_structured_info(qa):
    
    query = """ Extract the following fields from the document:
            - Author Name
            - Publication Date
            - Title
            - Abstract
            - Keywords
            - Summary
            
            Return as JSON."""
    # You can modify the query to extract specific information
    # based on the structure of your PDF documents

    return qa.run(query)