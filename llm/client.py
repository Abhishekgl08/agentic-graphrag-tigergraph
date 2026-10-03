import os
from openai import OpenAI
from app.exceptions import EmbeddingError


class AnswerGenerationClient:
    def __init__(self, model: str, client=None): self.model, self._client = model, client
    @property
    def client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"): raise EmbeddingError("OpenAI answer service is not configured")
            self._client = OpenAI()
        return self._client
    def answer(self, instructions: str, user_input: str) -> str:
        response = self.client.responses.create(model=self.model, instructions=instructions, input=user_input, store=False)
        if not response.output_text.strip(): raise EmbeddingError("Answer model returned no text")
        return response.output_text.strip()
