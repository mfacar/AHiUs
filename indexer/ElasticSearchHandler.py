import os
from typing import Dict, Any, Callable, Union

from elasticsearch import Elasticsearch
from elasticsearch import helpers

es = Elasticsearch(
    hosts=[{'host': "127.0.0.1", 'port': 9300}],
    use_ssl=False,
    verify_certs=False
)

# _map_to_action: Callable[[Elasticsearch], Dict[Union[str, Any], Union[str, Any], str]] = lambda item: {
#     "_index": 'idx_story',
#     '_op_type': item['action']['type'],
#     '_type': 'story',
#     item['action']['property']: item['data']}


class ElasticSearchWriter:
    @staticmethod
    def create(index, doc_type, body):
        return es.index(index, doc_type, body)

    @staticmethod
    def update(index, doc_type, body, id):
        return es.update(index, doc_type, body, id)

    @staticmethod
    def search(index, doc_type):
        return es.search(index=index, doc_type=doc_type)

    # @staticmethod
    # def bulk(items):
    #     return helpers.bulk(es, list(map(_map_to_action, items, op)))
