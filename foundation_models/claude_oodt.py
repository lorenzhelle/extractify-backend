import json
from typing import Union

import anthropic

from foundation_models.chat_openai import (
    AIModelType,
    get_api_key,
    get_model_name,
)


class ClaudeOODT:
    temperature: float
    system_prompt: Union[str, None] = None
    functions: list[any] = []
    model: AIModelType
    prompt: str

    def __init__(
        self,
        temperature=0.7,
        system_prompt: Union[str, None] = None,
        model: AIModelType = AIModelType.GPT4_TURBO,
    ):
        api_key = get_api_key(model)

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt

    async def generate_response(self, prompt: str) -> object:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "Du bist ein KI-Assistent, der dabei hilft, Suchanfragen zu klassifizieren und zu erkennen, ob sie in den Bereich 'Out-of-Domain' fallen."
        )
        model = get_model_name(model=self.model)

        response = await self.client.messages.create(
            model=model,
            system=system_message,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        # Extract and return the structured response as a JSON object
        try:
            structured_output = json.loads(response.content[0].text)
        except (json.JSONDecodeError, KeyError):
            print("Invalid response format", response.content[0].text)
            structured_output = {"error": "Invalid response format"}

        return structured_output
