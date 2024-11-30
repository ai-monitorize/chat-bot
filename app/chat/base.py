from typing import List


class ChatMessage(dict):
    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content


class UserMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='user', content=content)


class AssistantMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='assistant', content=content)


class SystemMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='system', content=content)


class Chat:
    def __init__(self, messages: List[ChatMessage]) -> None:
        self.messages = messages

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
