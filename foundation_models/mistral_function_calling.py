import json
from typing import Union

from foundation_models.chat_openai import (
    AIModelType,
    get_api_key,
    get_model_name,
)
from models.models import FilterGeneratorOutput
from mistralai.client import MistralClient


class MistralFunctionCalling:
    temperature: float
    system_prompt: Union[str, None] = None
    functions: list[any] = []
    model: AIModelType

    def __init__(
        self,
        temperature=0.7,
        functions=[],
        system_prompt: Union[str, None] = None,
        model: AIModelType = AIModelType.GPT4_TURBO,
    ):
        api_key = get_api_key(model)

        self.client = MistralClient(api_key=api_key)
        self.temperature = temperature
        self.model = model
        self.functions = functions
        self.system_prompt = system_prompt

    def generate_response(self, prompt: str) -> list[FilterGeneratorOutput]:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )
        model = get_model_name(model=self.model)

        print(self.functions)

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
            tool_choice="any",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "entity_linking",
                        "description": "Extrahiere die passenden Werte für die Filter aus der Konversation",
                        "parameters": self.functions[0]["parameters"],
                    },
                }
            ],
        )

        tool_output = response.choices[0].message.tool_calls[0].function.arguments

        if type(tool_output) is str:
            try:
                tool_output = json.loads(tool_output)
            except json.JSONDecodeError:
                tool_output = tool_output

        return tool_output

    def process_filter_data(self, tool_output):
        data = []

        # convert data to FilterGeneratorOutput
        for attr in tool_output:
            filter_data = tool_output[attr]

            # replace unknown token with Null if it is a string
            if type(filter_data) is str:
                filter_data = filter_data.replace("<UNKNOWN>", "null")
                # convert string to json
                try:
                    filter_data = json.loads(filter_data)
                except json.JSONDecodeError:
                    filter_data = filter_data

            if type(filter_data) is bool:
                continue

            if type(filter_data) is list:
                values = filter_data
                data.append(FilterGeneratorOutput(id=attr, values=values))
                continue

            if filter_data.get("max") is not None or filter_data.get("min") is not None:
                data.append(
                    FilterGeneratorOutput(
                        id=attr,
                        maximum=filter_data.get("max"),
                        minimum=filter_data.get("min"),
                        values=[],
                    )
                )
                continue

            if filter_data.get("values") is not None:
                print("handle values")
                # is discrete filter
                values = filter_data.get("values")
                # handle if values is a string instead of a list
                if type(values) is str:
                    values = [values]

                data.append(FilterGeneratorOutput(id=attr, values=values))

        return data
