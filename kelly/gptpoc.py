from typing import List

import pymongo
import together
from tqdm import tqdm

TOGETHER_API_KEY = '62584312b992b79e3e76c031ff115c6a10e72ab9d222a06266b7c2c55a6961e6'
MONGODB_URI = "mongodb+srv://hackers:aicampers@hackathon.zquobpp.mongodb.net/?retryWrites=true&w=majority&appName=hackathon"
together.api_key = TOGETHER_API_KEY
client = pymongo.MongoClient(MONGODB_URI)


# Step 2: Set up the embedding creation function
def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to each input text.
    """
    together_client = together.Together(api_key=TOGETHER_API_KEY)
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]


# Choose your embedding model
embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval'  # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval'  # define your embedding field name.
NUM_DOC_LIMIT = 200  # the number of documents you will process and generate embeddings.

# Test the embedding function
sample_output = generate_embeddings(["This is a test."], embedding_model_string)
print(f"Embedding size is: {str(len(sample_output[0]))}")

# Step 3: Create and store embeddings
db = client.sample_airbnb
collection_airbnb = db.listingsAndReviews

keys_to_extract = ["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access",
                   "interaction", "house_rules", "property_type", "room_type", "bed_type", "minimum_nights",
                   "maximum_nights", "accommodates", "bedrooms", "beds"]

for doc in tqdm(collection_airbnb.find({"summary": {"$exists": True}}).limit(NUM_DOC_LIMIT),
                desc="Document Processing"):
    extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
    if vector_database_field_name not in doc:
        doc[vector_database_field_name] = generate_embeddings([extracted_str], embedding_model_string)[0]
    collection_airbnb.replace_one({'_id': doc['_id']}, doc)

# Step 4: Create a vector search index in Atlas
# Follow the MongoDB Atlas Vector Search Index creation steps on your Atlas account page
# The JSON config for your embeddings:
"""
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding_together_m2-bert-8k-retrieval",
      "numDimensions": 768,
      "similarity": "dotProduct"
    }
  ]
}
"""

# Step 5: Retrieve
query = "apartment with a great view near a coast or beach for 4 people"
query_emb = generate_embeddings([query], embedding_model_string)[0]

results = collection_airbnb.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_emb,
            "path": vector_database_field_name,
            "numCandidates": 100,
            "limit": 10,
            "index": "SemanticSearch",
        }
    }
])
results_as_dict = {doc['name']: doc for doc in results}

print(f'From your query "{query}", the following airbnb listings were found:\n')
print("\n".join([str(i + 1) + ". " + name for (i, name) in enumerate(results_as_dict.keys())]))

# Step 6: Augment and Generate
your_task_prompt = (
    "From the given airbnb listing data, choose an apartment with a great view near a coast or beach for 4 people to stay for 4 nights. "
    "I want the apartment to have easy access to good restaurants. "
    "Tell me the name of the listing and why this works for me."
)

listing_data = ""
for doc in results_as_dict.values():
    listing_data += f"Listing name: {doc['name']}\n"
    for (k, v) in doc.items():
        if not (k in keys_to_extract) or ("embedding" in k): continue
        if k == "name": continue
        listing_data += k + ": " + str(v) + "\n"
    listing_data += "\n"

augmented_prompt = (
    "airbnb listing data:\n"
    f"{listing_data}\n\n"
    f"{your_task_prompt}"
)

generative_model_string = 'meta-llama/Llama-3-70b-chat-hf'

if __name__ == '__main__':
    response = together.Complete.create(
        prompt=augmented_prompt,
        model=generative_model_string,
        max_tokens=512,
        temperature=0.8,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
        stop=None,  # Add stop sequences if needed
    )

    print(response["output"]["choices"][0]["text"])
