from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Create a conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

def chat_with_doc():
    print("Welcome to the Conversational RAG system! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break
        
        result = qa({"question": user_input})
        print(f"AI: {result['answer']}")

if __name__ == "__main__":
    chat_with_doc()
