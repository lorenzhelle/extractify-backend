# Setup

## Create & setup new environment using conda & poetry

```bash
conda create -n entity-linking python=3.10

conda activate entity-linking
poetry install

```

## Start FAST api

PYTHONPATH=/Users/lorenz/Repos/extractify/backend python api/main.py
`cd backend/app`

Run the server:

`uvicorn main:app --reload`

## API Keys

A valid API-key is needed for GPT3.5 and GPT4 deployments on Azure. Currently the deployment of Azure Ressource owned by IncAI is used, the api-key is taken from `API_KEY_AZURE` in `.env` file (not checked in). See `backend/app/secret/openai.py` for details. To add you own API Key for a deployment, create a .env file and follow the .env-example file.

## Start MeiliSearch

The Backend uses MeiliSearch for search functionality. To start MeiliSearch, run the following command:

```
docker run -it --rm \
  -p 7700:7700 \
  -v $(pwd)/meili_data:/meili_data \
  getmeili/meilisearch:v1.4
```

Open MeiliSearch dashboard at http://localhost:7700/

### Index Product Data

Next step is to create an index and index the product data. To do this, follow the instruction in the `backend/meilisearch/meilisearch.ipynb` file.

### Use Deployed MeiliSearch

To use a deployed MeiliSearch, set the `MEILI_URL` in the `.env` file to the deployed MeiliSearch URL.

# Docker Local

You can also run the backend in a docker container.

## Build

```
docker build -t backend .
```

And Run with

```
docker run -p 80:80 backend
```
