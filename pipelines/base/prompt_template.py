from langchain import PromptTemplate

chatbot_template = """
{history}
Human: {human_input}
AI:"""
chatbot_prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=chatbot_template
)