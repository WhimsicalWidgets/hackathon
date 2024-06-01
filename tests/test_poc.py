import pytest

from kelly.mongo import Prompt, PromptView
from kelly.poc import generate_embeddings, app, mongo_insert_prompt, init_mongo, search_prompt_content


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_search_with_query(client):
    query = 'test'
    expected_results = [{'content': 'example content'}]

    response = client.get('/search', query_string={'query': query})
    json_data = response.get_json()

    assert response.status_code == 200
    assert 'results' in json_data
    assert json_data['results'] == expected_results


def test_search_without_query(client):
    response = client.get('/search')
    json_data = response.get_json()

    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == 'No query provided'


def test_generate_embeddings():
    print(generate_embeddings('hello'))


# /Users/kellyjohnson/kelly/hackathon/hackathon-backend/tests
@pytest.mark.asyncio
async def test_insert():
    await init_mongo()
    await mongo_insert_prompt('Prompt!')
    print(await Prompt.find_one().to_list())


@pytest.mark.asyncio
async def test_find():
    await init_mongo()
    # .project(PromptView)
    print(await Prompt.find().to_list())


@pytest.mark.asyncio
async def test_search():
    await init_mongo()
    results = await search_prompt_content('prompt')
    print(results)
