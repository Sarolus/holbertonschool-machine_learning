#!/usr/bin/env python3

def top_students(mongo_collection):
    """
        Returns all students storted by average score

        Args:
            mongo_collection: pymongo collection object

        Returns:
            list of students sorted by average score
    """

    return mongo_collection.aggregate([
        {
            '$project': {
                '_id': 0,
                'name': 1,
                'averageScore': {
                    '$avg': '$topics.score'
                }
            }
        },
        {
            '$sort': {
                'averageScore': -1
            }
        }
    ])
