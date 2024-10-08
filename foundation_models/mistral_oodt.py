import json
from typing import Union

from foundation_models.chat_openai import (
    AIModelType,
    get_api_key,
    get_model_name,
)
from mistralai.client import MistralClient


class MistralOOTD:
    temperature: float
    system_prompt: Union[str, None] = None
    functions: list[any] = []
    model: AIModelType

    def __init__(
        self,
        temperature=0.7,
        system_prompt: Union[str, None] = None,
        model: AIModelType = AIModelType.MISTRAL_MIXTRAL_8x22B,
    ):
        api_key = get_api_key(model)

        self.client = MistralClient(api_key=api_key)
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

        response = self.client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        # Extract and return the structured response as a JSON object
        try:
            structured_output = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError):
            print("Invalid response format", response.choices[0].message.content)
            structured_output = {"error": "Invalid response format"}

        return structured_output
