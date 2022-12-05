#!/usr/bin/env python3

def update_topics(mongo_collection, name, topics):
    """
        Changes all topics of a school document based on the name

        Args:
            mongo_collection: pymongo collection object
            name: school name to update
            topics: list of topics approached in the school

        Returns:
            the number of documents updated
    """
    return mongo_collection.update_many({"name":
                                            name}, {"$set": {"topics": topics}}).modified_count
