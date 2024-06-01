from datetime import datetime
from typing import Any, List, Optional, Union

from beanie import Document
from pydantic import BaseModel, Field
from pymongo import MongoClient


class VersionHistory(BaseModel):
    prompt_id: str
    timestamp: datetime
    prompt: str


class PromptView(BaseModel):
    content: str


class Prompt(Document):
    content: str
    content_embeddings: List[float]

    class Settings:
        name = "Prompts"


class FilterCondition(BaseModel):
    field: str
    operator: str
    value: Any


class LogicalCondition(BaseModel):
    and_: Optional[List['QueryFilter']] = Field(None, alias='$and')
    or_: Optional[List['QueryFilter']] = Field(None, alias='$or')
    not_: Optional['QueryFilter'] = Field(None, alias='$not')
    nor_: Optional[List['QueryFilter']] = Field(None, alias='$nor')


class QueryFilter(BaseModel):
    conditions: List[Union[FilterCondition, LogicalCondition]]


class Projection(BaseModel):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class Sort(BaseModel):
    field: str
    order: int  # 1 for ascending, -1 for descending


class UpdateOperation(BaseModel):
    field: str
    operator: str
    value: Any


class Query(BaseModel):
    filter: Optional[QueryFilter] = None
    projection: Optional[Projection] = None
    sort: Optional[List[Sort]] = None
    limit: Optional[int] = None
    skip: Optional[int] = None


class Update(BaseModel):
    operations: List[UpdateOperation]


class MongoQuery(BaseModel):
    query: Query
    update: Optional[Update] = None


class VectorSearchQuery(BaseModel):
    index: str = Field(default="vector_index")
    path: str = Field(default="content_embeddings")
    # filter: Optional[dict] = {}
    queryVector: List[float]
    numCandidates: int = Field(default=200)
    limit: int = Field(default=10)


class MongoDB:
    def __init__(self, uri: str, database: str, collection: str):
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def vector_search(self, query: VectorSearchQuery):
        agg = [
            {
                '$vectorSearch': {
                    'index': query.index,
                    'path': query.path,
                    'filter': query.filter,
                    'queryVector': query.queryVector,
                    'numCandidates': query.numCandidates,
                    'limit': query.limit
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'content': 1,
                    'score': {
                        '$meta': 'vectorSearchScore',
                    },
                },
            },
        ]
        result = self.collection.aggregate(agg)
        return list(result)
