#!/usr/bin/env python3
"""
    Collection Listing with Specified Topic Module
"""


def schools_by_topic(mongo_collection, topic):
    """
        Returns the list of school having a specific topic

        Args:
            mongo_collection: pymongo collection object
            topic: topic searched

        Returns:
            list of school having the topic
    """

    return [x for x in mongo_collection.find({"topics": topic})]
