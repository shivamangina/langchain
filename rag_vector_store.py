from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load and preprocess the document
loader = TextLoader('path/to/your/document.txt')
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create and persist the vector store
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the language model
llm = OpenAI(temperature=0)

# Create a question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Function to query the document
def query_document(question):
    result = qa_chain({"query": question})
    return result["result"]

# Example usage
if __name__ == "__main__":
    question = "What is the main topic of the document?"
    answer = query_document(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
