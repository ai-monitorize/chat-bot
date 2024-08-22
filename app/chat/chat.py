class ChatMessage(dict):
    def __init__(self, role: str, content: str) -> None:
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content


class Chat:
    def __init__(self) -> None:
        self.messages = []

    def add_message(self, message: ChatMessage, index: int) -> None:
        if index >= len(self.messages):
            self.messages.append(message)
        else:
            self.messages[index] = message

    def add_user_message(self, message_text: str) -> None:
        self.messages.append(ChatMessage(role='user', content=message_text))

    def add_assistant_message(self, message_text: str) -> None:
        self.messages.append(ChatMessage(role='assistant', content=message_text))

    def add_system_message(self, message_text: str) -> None:
        self.messages.append(ChatMessage(role='system', content=message_text))
