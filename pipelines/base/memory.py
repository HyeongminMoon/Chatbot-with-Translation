from langchain.memory import ConversationBufferWindowMemory
from collections import defaultdict
from functools import partial


class ChatbotMemory:
    memory = defaultdict(partial(ConversationBufferWindowMemory, k=2))

    @classmethod
    def save_context(cls, user_id: str, user_input: str, output: str):
        cls.memory[hash(user_id)].save_context({"input": user_input}, {"output": output})

    @classmethod
    def load_context(cls, user_id: str):
        return cls.memory[hash(user_id)].load_memory_variables({})

    @classmethod
    def clear_memory(cls, user_id: str):
        cls.memory[hash(user_id)] = ConversationBufferWindowMemory(k=2)