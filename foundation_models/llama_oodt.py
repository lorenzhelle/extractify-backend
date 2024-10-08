import json
from typing import Union

import replicate
from mistralai.client import MistralClient

from foundation_models.chat_openai import AIModelType, get_api_key, get_model_name


class LlamaOOTD:
    temperature: float
    system_prompt: Union[str, None] = None
    functions: list[any] = []
    model: AIModelType

    def __init__(
        self,
        temperature=0.7,
        system_prompt: Union[str, None] = None,
        model: AIModelType = AIModelType.GPT4_TURBO,
    ):
        api_key = get_api_key(model)

        self.client = MistralClient(api_key=api_key)
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt

    async def generate_response(self, prompt: str) -> object:
        model = get_model_name(model=self.model)

        response = ""

        try:
            # The meta/meta-llama-3-8b model can stream output as it's running.
            for event in replicate.stream(
                model,
                input={
                    "top_p": 0.9,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "presence_penalty": 1.15,
                },
            ):
                response += str(event)
        except Exception as e:
            print("error", e)
        # Extract and return the structured response as a JSON object
        try:
            print("response", response)
            structured_output = json.loads(response)
        except (json.JSONDecodeError, KeyError) as error:
            print("Invalid response format", response, error)

            structured_output = {"error": "Invalid response format"}

        return structured_output
