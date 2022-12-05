#!/usr/bin/env python3
"""
    Collection Listing Module
"""


def list_all(mongo_collection):
    """
        Lists all documents in a collection

        Args:
            mongo_collection: pymongo collection object

        Returns:
            list of documents in collection
    """

    if mongo_collection.count_documents({}) == 0:
        return []

    return [x for x in mongo_collection.find()]
