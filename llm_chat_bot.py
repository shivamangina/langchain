from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.7)

# Create a conversation memory
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

def chat_with_bot():
    print("Welcome to the LLM Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = conversation.predict(input=user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat_with_bot()
