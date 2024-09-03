from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to get a response from the LLM
def get_llm_response(question):
    response = chain.run(question)
    return response

# Example usage
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    answer = get_llm_response(user_question)
    print(f"Answer: {answer}")
