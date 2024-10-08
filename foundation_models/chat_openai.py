from enum import Enum
from typing import Generator, Union

from utils.secret.api_keys import (
    get_api_key_from_env_file,
)
from openai import AsyncAzureOpenAI
from openai import AzureOpenAI


class AIModelType(str, Enum):
    GPT3 = "GPT3"
    GPT4_TURBO = "GPT4_TURBO"
    GPT4_O = "GPT4_O"
    GPT4_O_MINI = "GPT4_O_MINI"
    CLAUDE_OPUS = "CLAUDE_OPUS"
    CLAUDE_SONNET = "CLAUDE_SONNET"
    MISTRAL_LARGE = "MISTRAL_LARGE"
    MISTRAL_SMALL = "MISTRAL_SMALL"
    MISTRAL_MIXTRAL_8x22B = "MISTRAL_MIXTRAL_8x22B"
    GOOGLE_GEMINI_PRO = "GOOGLE_GEMINI_PRO"
    LLAMA_3_8B = "LLAMA_3_8B"
    LLAMA_3_70B = "LLAMA_3_70B"


def get_api_base(model: AIModelType) -> str:
    if (
        model == AIModelType.GPT3
        or model == AIModelType.GPT4_TURBO
        or model == AIModelType.GPT4_O
        or model == AIModelType.GPT4_O_MINI
    ):
        print("no api base needed")
        return None
    else:
        raise ValueError("Invalid AI model")


def get_api_key(model: AIModelType) -> str:
    if (
        model == AIModelType.GPT3
        or model == AIModelType.GPT4_TURBO
        or model == AIModelType.GPT4_O
        or model == AIModelType.GPT4_O_MINI
    ):
        return get_api_key_from_env_file("API_KEY_OPEN_AI")
    if model == AIModelType.CLAUDE_OPUS or model == AIModelType.CLAUDE_SONNET:
        print("get api key for model", model)
        print("model", model)
        return get_api_key_from_env_file("API_KEY_CLAUDE")
    if (
        model == AIModelType.MISTRAL_LARGE
        or model == AIModelType.MISTRAL_MIXTRAL_8x22B
        or model == AIModelType.MISTRAL_SMALL
    ):
        return get_api_key_from_env_file("API_KEY_MISTRAL")
    elif model == AIModelType.GOOGLE_GEMINI_PRO:
        return get_api_key_from_env_file("API_KEY_GOOGLE")
    elif model == AIModelType.LLAMA_3_8B or model == AIModelType.LLAMA_3_70B:
        return "dummy"
    else:
        raise ValueError(f"No API key found for {model}")


def get_model_name(model: AIModelType) -> str:
    if model == AIModelType.GPT3:
        return "gpt-3.5-turbo"
    elif model == AIModelType.GPT4_TURBO:
        return "gpt-4-turbo"
    elif model == AIModelType.GPT4_O:
        return "gpt-4o"
    elif model == AIModelType.GPT4_O_MINI:
        return "gpt-4o-mini"
    elif model == AIModelType.CLAUDE_OPUS:
        return "claude-3-opus-20240229"
    elif model == AIModelType.CLAUDE_SONNET:
        return "claude-3-sonnet-20240229"
    elif model == AIModelType.MISTRAL_LARGE:
        return "mistral-large-latest"
    elif model == AIModelType.MISTRAL_MIXTRAL_8x22B:
        return "open-mixtral-8x22b"
    elif model == AIModelType.MISTRAL_SMALL:
        return "mistral-small-latest"
    elif model == AIModelType.GOOGLE_GEMINI_PRO:
        return "models/gemini-1.5-pro-latest"
    elif model == AIModelType.LLAMA_3_8B:
        return "meta/meta-llama-3-8b"
    elif model == AIModelType.LLAMA_3_70B:
        return "meta/meta-llama-3-70b-instruct"
    else:
        raise ValueError("Invalid AI model type")


class ChatOpenAI:
    temperature: float
    system_prompt: Union[str, None] = None
    model: AIModelType = AIModelType.GPT4_TURBO

    def __init__(
        self,
        temperature=0.7,
        model: AIModelType = AIModelType.GPT3,
        system_prompt: Union[str, None] = None,
    ):
        self.openai = AsyncAzureOpenAI(
            azure_endpoint=get_api_base(model),
            api_key=get_api_key(model),
            api_version="2023-07-01-preview",
        )
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.model = model

    def generate_prompt(self) -> str:
        return ""

    async def generate_response(self, prompt: str) -> object:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )
        model = get_model_name(self.model)

        response = await self.openai.chat.completions.create(
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

        return response

    async def generate__stream_response(self, prompt: str) -> object:
        model = get_model_name(self.model)
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )

        response = self.openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in await response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                current_content = chunk.choices[0].delta.content
                print(current_content)
                yield current_content

    def generate__stream_response_sse(self, prompt: str) -> Generator:

        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )

        self.openai = AzureOpenAI(
            azure_endpoint=get_api_base(self.model),
            api_key=get_api_key(self.model),
            api_version="2023-07-01-preview",
        )

        model_name = get_model_name(self.model)
        openai_stream = self.openai.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )

        for chunk in openai_stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                current_content = chunk.choices[0].delta.content
                yield current_content

    def name(self) -> str:
        return f"ChatOpenAI with (temperature={self.temperature})"
