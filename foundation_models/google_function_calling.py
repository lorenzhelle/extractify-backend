import json
from typing import Union

import proto

from foundation_models.chat_openai import (
    AIModelType,
    get_api_key,
    get_model_name,
)
from models.models import FilterGeneratorOutput
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)


class GoogleFunctionCalling:
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

        self.temperature = temperature
        self.model = model
        self.functions = functions
        self.system_prompt = system_prompt

    async def generate_response(self, prompt: str) -> list[FilterGeneratorOutput]:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )
        model_name = get_model_name(model=self.model)

        entity_linking_function = FunctionDeclaration(
            name="entity_linking",
            description="Extrahiere die Werte für die Filter aus der Konversation",
            parameters={
                "type": "object",
                "description": "Parameter für die Funktion",
                "properties": {
                    "Kategorie": {
                        "type": "string",
                        "description": "Kategorie des gewünschten Produkts.",
                        "enum": [
                            "Fernseher",
                            "Mobiltelefone",
                            "Tablets",
                            "Computer",
                            "Monitor",
                            "Laptops",
                            "Andere",
                        ],
                    },
                    "Bildschirmgroesse": {
                        "type": "array",
                        "description": "Bildschirmgroesse des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "gr\u00f6\u00dfer als 34 Zoll",
                                "9 Zoll",
                                "10 Zoll",
                                "11 Zoll",
                                "17 Zoll",
                                "24 Zoll",
                                "gr\u00f6\u00dfer als 19 Zoll",
                                "65 - 69 Zoll",
                                "gr\u00f6\u00dfer als 84 Zoll",
                                "75 - 79 Zoll",
                                "40 - 44 Zoll",
                                "12 Zoll",
                                "28 Zoll",
                                "34 Zoll",
                                "15 Zoll",
                                "kleiner als 12 Zoll",
                                "14 Zoll",
                                "16 Zoll",
                                "32 Zoll",
                                "30 - 34 Zoll",
                                "55 - 59 Zoll",
                                "13 Zoll",
                                "27 Zoll",
                                "50 - 54 Zoll",
                            ],
                        },
                    },
                    "Marke": {
                        "type": "array",
                        "description": "Marke des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Hisense",
                                "Lenovo",
                                "Sony",
                                "Philips",
                                "Huawei",
                                "Apple",
                                "AOC",
                                "Dell",
                                "Samsung",
                                "Acer",
                                "LG",
                                "Asus",
                                "Motorola",
                                "HP",
                                "Nokia",
                                "Xiaomi",
                            ],
                        },
                    },
                    "Displaytechnologie": {
                        "type": "array",
                        "description": "Displaytechnologie des gewünschten Produkts",
                        "items": {
                            "type": "string",
                            "enum": ["QLED", "IPS", "OLED"],
                        },
                    },
                    "Ausstattung": {
                        "type": "array",
                        "description": "Ausstattung des gewünschten Produkts",
                        "items": {
                            "type": "string",
                            "enum": [
                                "kabelloses Laden",
                                "Smart-TV",
                                "Stifteingabe",
                                "SSD-Festplatte",
                                "Ambilight",
                                "2 in 1 Convertible",
                                "5G",
                                "mobiles Internet",
                                "Android TV",
                                "Touch Display",
                                "Tastatur",
                                "Wifi",
                                "GPS",
                            ],
                        },
                    },
                    "Aufloesung": {
                        "type": "array",
                        "description": "Aufloesung des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": ["Ultra-HD-8K", "Ultra-HD ", "Ultra-HD-4K"],
                        },
                    },
                    "Farbe": {
                        "type": "array",
                        "description": "Farbe des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "silberfarben",
                                "weiß",
                                "rosa",
                                "blau",
                                "goldfarben",
                                "schwarz",
                            ],
                        },
                    },
                    "Bildwiederholfrequenz": {
                        "type": "array",
                        "description": "Bildwiederholfrequenz des gewünschten Produkts.",
                        "items": {"type": "string", "enum": ["100 Hz"]},
                    },
                    "Zubehoerfuer": {
                        "type": "array",
                        "description": "Zubehoerfuer des gewünschten Produkts.",
                        "items": {"type": "string", "enum": ["Gaming"]},
                    },
                    "Betriebssystem": {
                        "type": "array",
                        "description": "Betriebssystem des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": ["Android", "Windows"],
                        },
                    },
                    "Prozessor": {
                        "type": "array",
                        "description": "Prozessor des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Intel Core i3",
                                "M3 Pro",
                                "Intel",
                                "Intel Core i7",
                                "Intel Core i5",
                                "M1",
                                "Intel Core i9",
                                "AMD Ryzen 7",
                                "AMD Ryzen 5",
                                "M2",
                                "M2 Pro",
                                "M2 Max",
                                "M3",
                                "M3 Max",
                            ],
                        },
                    },
                    "Arbeitsspeicher": {
                        "type": "array",
                        "description": "Arbeitsspeicher des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "32 GB",
                                "64 GB",
                                "16 GB",
                                "8 GB",
                                "256 GB",
                            ],
                        },
                    },
                    "Speicherkapazitaet": {
                        "type": "array",
                        "description": "Speicherkapazitaet des gewünschten Produkts.",
                        "description": "Speicherkapazitaet die genannt wurden",
                        "items": {
                            "type": "string",
                            "enum": [
                                "512 GB",
                                "1 TB",
                                "2 TB",
                                "128 GB",
                                "256 GB",
                            ],
                        },
                    },
                    "Modellreihe": {
                        "type": "array",
                        "description": "Modellreihe des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Spin 3",
                                "MacBook Pro",
                                "iPad Air",
                                "Chromebook",
                                "iPhone 15",
                                "iPhone 14",
                                "Galaxy S24",
                                "Galaxy S23",
                                "iPad Pro",
                                "iPhone 12",
                                "iPhone 15 Pro",
                                "Galaxy S22",
                                "MacBook Air",
                                "MacBook",
                                "Galaxy",
                                "iPhone 13",
                                "A54",
                                "iPad",
                            ],
                        },
                    },
                    "Grafikkarte": {
                        "type": "array",
                        "description": "Grafikkarte des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "RTX 4060",
                                "RTX 4090",
                                "RTX 3070" "GeForce RTX 4070",
                                "GeForce RTX 4090",
                            ],
                        },
                    },
                    "Bildwiederholungsfrequenz": {
                        "type": "array",
                        "description": "Bildwiederholungsfrequenz des gewünschten Produkts.",
                        "items": {
                            "type": "string",
                            "enum": ["144 Hz", "240 Hz", "165 Hz", "120 Hz"],
                        },
                    },
                    "Art": {
                        "type": "array",
                        "description": "Art des gewünschten Produkts.",
                        "items": {"type": "string", "enum": ["curved"]},
                    },
                    "Reaktionszeit": {
                        "type": "array",
                        "description": "Reaktionszeit des gewünschten Produkts.",
                        "items": {"type": "string", "enum": ["bis 1 ms"]},
                    },
                    "Preis": {
                        "type": "object",
                        "description": "Preis des gewünschten Produkts. Preis ist angegegeben in Euro.",
                        "properties": {
                            "min": {
                                "type": "integer",
                                "description": "Mindest-Preis des gewünschten Produkts",
                                "minimum": 0,
                            },
                            "max": {
                                "type": "integer",
                                "description": "Maximal-Preis des gewünschten Produkts",
                                "minimum": 0,
                            },
                            "noSpecificUserPreference": {
                                "type": "boolean",
                                "description": "True wenn der Kunde keine bestimmte Präferenz für den Preis hat, sonst False",
                            },
                        },
                    },
                },
            },
        )

        entity_linking_tool = Tool(
            function_declarations=[entity_linking_function],
        )

        model = GenerativeModel(
            model_name="gemini-1.5-pro-001",
            system_instruction=system_message,
        )

        response = model.generate_content(prompt, tools=[entity_linking_tool])

        response.candidates[0].content

        params = {}
        for key, value in (
            response.candidates[0].content.parts[0].function_call.args.items()
        ):
            params[key] = value
            print(key, value)
        params

        print(params)
        data = []

        # convert data to FilterGeneratorOutput
        for attr in params:
            filter_data = params[attr]

            if filter_data == "unknown":
                filter_data = None
                continue

            if type(filter_data) is proto.marshal.collections.maps.MapComposite:
                # convert proto map to dict
                print("convert to dict", filter_data)
                filter_data = dict(filter_data)

            # replace unknown token with Null if it is a string
            if type(filter_data) is str:
                filter_data = filter_data.replace("unknown", "null")
                # convert string to json
                filter_data = json.loads(filter_data)

            if type(filter_data) is bool:
                continue

            if (
                type(filter_data) is list
                or type(filter_data)
                is proto.marshal.collections.repeated.RepeatedComposite
            ):
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

                print("append data", values)
                data.append(FilterGeneratorOutput(id=attr, values=values))

        return data

    async def generate_response_generic(self, prompt: str) -> dict:
        system_message = (
            self.system_prompt
            if self.system_prompt is not None
            else "You are an AI assistant that helps people find information."
        )
        model_name = get_model_name(model=self.model)

        function = self.functions[0]

        function = FunctionDeclaration(
            name=function["name"],
            description=function["description"],
            parameters=function["parameters"],
        )

        entity_linking_tool = Tool(
            function_declarations=[function],
        )

        model = GenerativeModel(
            model_name="gemini-1.5-pro-001",
            system_instruction=system_message,
        )

        response = model.generate_content(prompt, tools=[entity_linking_tool])

        print("raw response", response)

        print(
            "response.candidates[0].content.args",
            response.candidates[0].content,
        )

        params = {}
        for key, value in (
            response.candidates[0].content.parts[0].function_call.args.items()
        ):
            params[key] = value
        params

        return params
