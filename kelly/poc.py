import asyncio
from typing import List

import together
from beanie import init_beanie
from flask import Flask, request, jsonify
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from quart import Quart

from kelly.mongo import Prompt, VectorSearchQuery, PromptView

TOGETHER_API_KEY = '62584312b992b79e3e76c031ff115c6a10e72ab9d222a06266b7c2c55a6961e6'
MONGODB_URI = "mongodb+srv://hackers:aicampers@hackathon.zquobpp.mongodb.net/?retryWrites=true&w=majority&appName=hackathon"
mongodb_client = MongoClient(MONGODB_URI)


# vector_store = MongoDBAtlasVectorSearch(mongodb_client)


async def mongo_insert_prompt(content):
    embeddings = generate_embeddings([content])[0]
    prompt = Prompt(content=content, content_embeddings=embeddings)
    await prompt.insert()


app = Quart(__name__)


@app.before_request
def initialize():
    if not hasattr(app, 'db_initialized'):
        asyncio.run(init_mongo())
        app.db_initialized = True


@app.route('/search', methods=['GET'])
async def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    results = await search_prompt_content(query)
    return jsonify({'results': results})


async def init_mongo():
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URI)

    # Select the database
    db = client.Hackathon_DB

    # Initialize beanie with the Prompt document class
    await init_beanie(database=db, document_models=[Prompt])


def generate_embeddings(input_texts: List[str]) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """

    embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval'
    together_client = together.Together(api_key=TOGETHER_API_KEY)
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=embedding_model_string,
    )
    return [x.embedding + x.embedding for x in outputs.data]


async def search_prompt_content(content) -> List[PromptView]:
    embeddings = generate_embeddings([content])[0]
    query = VectorSearchQuery(queryVector=embeddings)
    agg = [
        {
            '$vectorSearch': query.model_dump(),
        },
        {
            '$project': {
                '_id': 0,
                'content': 1,
                'score': {
                    '$meta': 'vectorSearchScore',
                },
            },
        }
    ]
    print(agg)
    size_aggregate = [{'$sample': {'size': 1}}]
    return await Prompt.aggregate(size_).to_list()


if __name__ == '__main__':
    app.run(debug=True)
