import json
import re
from typing import Union

import replicate


from foundation_models.chat_openai import AIModelType, get_api_key, get_model_name
from models.models import FilterGeneratorOutput


entity_linking_function_small = {
    "name": "entity_linking",
    "description": "Extrahiere die Werte für die Filter aus der Konversation",
    "parameters": {
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
        },
    },
}

entity_linking_function = {
    "name": "entity_linking",
    "description": "Extrahiere die Werte für die Filter aus der Konversation",
    "parameters": {
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
}


class LlamaFunctionCalling:
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

        response = ""

        print("self.functions", self.functions[0])

        toolPrompt = f"""
            You have access to the following functions:

            Use the function '{self.functions[0]["name"]}' to '{self.functions[0]["description"]}':
            {json.dumps(self.functions[0])}

            If you choose to call a function ONLY reply in the following format with no prefix or suffix:

            <function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

            Reminder:
            - Function calls MUST follow the specified format, start with <function= and end with </function>
            - Required parameters MUST be specified
            - Only call one function at a time
            - Put the entire function call reply on one line
            - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls

            """

        try:
            # The meta/meta-llama-3-8b model can stream output as it's running.
            for event in replicate.stream(
                model_name,
                input={
                    "top_p": 0.9,
                    "prompt": prompt + toolPrompt,
                    "min_tokens": 100,
                    "temperature": self.temperature,
                    "presence_penalty": 1.15,
                },
            ):
                response += str(event)
        except Exception as e:
            print("error", e)

        if response is None:
            return None

        # try:
        parsed_response = parse_tool_response(response)["arguments"]

        return parsed_response
        data = []

        # convert data to FilterGeneratorOutput
        for attr in parsed_response:
            print("attr", attr)
            filter_data = parsed_response[attr]

            if filter_data == "unknown":
                filter_data = None
                continue

            # replace unknown token with Null if it is a string
            if type(filter_data) is str:
                filter_data = filter_data.replace("unknown", "null")
                # convert string to json
                filter_data = [filter_data]

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

                print("append data", values)
                data.append(FilterGeneratorOutput(id=attr, values=values))

        return data
        # except Exception as e:
        #     print("error parsing response", e)
        #     return []


def parse_tool_response(response: str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)

    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {
                "function": function_name,
                "arguments": args,
            }
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None
    return None
