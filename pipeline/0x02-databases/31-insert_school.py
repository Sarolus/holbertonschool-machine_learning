#!/usr/bin/env python3
"""
    New Entry in Collection Insertion Module
"""


def insert_school(mongo_collection, **kwargs):
    """
        Inserts a new document in a collection based on kwargs

        Args:
            mongo_collection: pymongo collection object
            **kwargs: will be used to create a new document

        Returns:
            the new _id
    """

    return mongo_collection.insert_one(kwargs).inserted_id
