import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from entity_linking import EntityLinking
from foundation_models.chat_openai import AIModelType
from foundation_models.claude_oodt import ClaudeOODT
from foundation_models.mistral_oodt import MistralOOTD
from foundation_models.llama_oodt import LlamaOOTD
from foundation_models.openai_oodt import ChatOpenAIOutOfDomainDetection

# load_and_export.py
from dotenv import load_dotenv
import os

app = FastAPI()

# Load the environment variables from .env file
load_dotenv()


class QueryRequest(BaseModel):
    query: str
    model: AIModelType
    domain: str


class FilterRequest(BaseModel):
    message: str
    schema: str
    model: AIModelType
    domain_definition: Optional[str] = None


class FilterResponse(BaseModel):
    filter_generator_output: List[Dict[str, Any]]
    recognized_filters: List[Dict[str, Any]]


def generate_target_schema(
    input_schema: Dict[str, Any], model: AIModelType
) -> Dict[str, Any]:
    target_schema = {
        "name": "entity_linking",
        "description": "Extrahiere die passenden Werte für die Filter aus der Anfrage",
        "parameters": {
            **input_schema,
            "description": "Parameter für die Funktion",
        },
    }

    return target_schema


@app.post("/entity-linking")
async def entity_linking(request: FilterRequest):
    try:
        input_schema = json.loads(request.schema)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON schema format: {str(e)}"
        )

    try:

        message = request.message

        if request.domain_definition is not None:
            message += (
                f"\n\nDomain definition: {request.domain_definition} \n"
                + request.message
            )

        # target_schema = generate_target_schema(input_schema, request.model)
        target_schema = input_schema
        llm_module = EntityLinking(schema=target_schema, model=request.model)

        if (
            request.model == AIModelType.MISTRAL_LARGE
            or request.model == AIModelType.MISTRAL_MIXTRAL_8x22B
            or request.model == AIModelType.MISTRAL_SMALL
        ):
            filter_generator_output = llm_module.generate_sync(conversation=message)
        else:
            filter_generator_output = await llm_module.generate_response_generic(
                conversation=message
            )
    except Exception as e:
        print(f"Error in entity-linking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return filter_generator_output


@app.post("/check_domain")
async def check_domain(request: QueryRequest):
    prompt = f"""
    Beantworte die Frage, ob diese Anfrage in deine Beratungsdomäne fällt oder nicht. Bedenke dabei, dass du nur für die Verkaufsberatung von {request.domain} zuständig bist.

    Query: "{request.query}"

    Gib die Antwort in folgender JSON-Struktur zurück:
    
    {{
    "query": "{request.query}",
    "outOfDomain": true/false
    }}
    """
    try:
        if request.model in [AIModelType.CLAUDE_OPUS, AIModelType.CLAUDE_SONNET]:
            chat_model = ClaudeOODT(model=request.model)
        elif request.model in [AIModelType.MISTRAL_LARGE, AIModelType.MISTRAL_SMALL]:
            chat_model = MistralOOTD(model=request.model)
        elif request.model in [AIModelType.LLAMA_3_8B, AIModelType.LLAMA_3_70B]:
            chat_model = LlamaOOTD(model=request.model)
        elif request.model in [
            AIModelType.GPT3,
            AIModelType.GPT4_TURBO,
            AIModelType.GPT4_O_MINI,
            AIModelType.GPT4_O,
        ]:
            chat_model = ChatOpenAIOutOfDomainDetection(model=request.model)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        response = await chat_model.generate_response(prompt=prompt)
        print(response)
        return {"inDomain": not response.get("outOfDomain", False)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
