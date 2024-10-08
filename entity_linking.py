from foundation_models.chat_openai import AIModelType

from foundation_models.claude_function_calling import ClaudeFunctionCalling
from foundation_models.google_function_calling import GoogleFunctionCalling
from foundation_models.llama_function_calling import LlamaFunctionCalling
from foundation_models.mistral_function_calling import MistralFunctionCalling
from foundation_models.openai_function_calling import (
    ChatOpenAIFunctionCalling,
)
from langchain.prompts.prompt import PromptTemplate

from models.models import FilterGeneratorOutput


template = """
Gegeben ist folgende Anfrage in einem E-Commerce Shop.

Query:
{query}

Extrahiere die gewünschten Filterwerte aus der Konversation, nutze dabei nur die zur Verfügung stehenden Filterwerte. Bedenke aber auch implizite Filterwerte, die nicht explizit genannt werden,
aber aus dem Kontext hervorgehen und helfen dem Kunden das passende Produkt zu finden.

Denke dabei Schritt für Schritt vor und nutze die Funktionen, die dir zur Verfügung stehen. Erfinde keine neuen Werte in den Filter Werten.
Wenn du für das Gespräch keine passendes Filter Werte findest, gib ein leeres Array zurück.
"""


function_calling_ner_tempalte = PromptTemplate.from_template(template)


class EntityLinking:
    functions_schema: str
    model: AIModelType

    def __init__(self, schema: str, model: AIModelType) -> None:
        super().__init__()
        system_prompt = "Only use the functions you have been provided with. Only use existing filter values. If you dont find filter values that match the conversation, return an empty array."

        if model == AIModelType.CLAUDE_OPUS or model == AIModelType.CLAUDE_SONNET:
            self.llm = ClaudeFunctionCalling(
                temperature=0.0,
                system_prompt=system_prompt,
                functions=[schema],
                model=model,
            )
        elif (
            model == AIModelType.MISTRAL_LARGE
            or model == AIModelType.MISTRAL_MIXTRAL_8x22B
            or model == AIModelType.MISTRAL_SMALL
        ):
            print("using mistral")
            self.llm = MistralFunctionCalling(
                temperature=0.0,
                system_prompt=system_prompt,
                functions=[schema],
                model=model,
            )
        elif model == AIModelType.GOOGLE_GEMINI_PRO:
            print("using google")
            self.llm = GoogleFunctionCalling(
                temperature=0.0,
                system_prompt=system_prompt,
                functions=[schema],
                model=model,
            )
        elif model == AIModelType.LLAMA_3_8B or model == AIModelType.LLAMA_3_70B:
            self.llm = LlamaFunctionCalling(
                temperature=0.0,
                system_prompt=system_prompt,
                functions=[schema],
                model=model,
            )
        else:
            self.llm = ChatOpenAIFunctionCalling(
                temperature=0.0,
                system_prompt=system_prompt,
                functions=[schema],
                model=model,
            )

    async def generate_async(self, conversation: str) -> list[FilterGeneratorOutput]:
        prompt = function_calling_ner_tempalte.format(query=conversation)
        response = await self.llm.generate_response(prompt=prompt)

        return response

    def generate_sync(self, conversation: str) -> list[FilterGeneratorOutput]:
        prompt = function_calling_ner_tempalte.format(query=conversation)
        response = self.llm.generate_response(prompt=prompt)

        return response

    def generate_response_generic(self, conversation: str) -> object:
        prompt = function_calling_ner_tempalte.format(query=conversation)
        response = self.llm.generate_response_generic(prompt=prompt)

        return response
