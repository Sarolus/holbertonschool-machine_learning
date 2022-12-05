#!/usr/bin/env python3
"""
    List 10 most present IPs in the collection Nginx
"""

from pymongo import MongoClient
import urllib

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    database = client.logs
    collection = database.nginx

    print("{} logs".format(collection.count_documents({})))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        print("\tmethod {}: {}".format(method, collection.count_documents({"method": method})))

    print("{} status check".format(collection.count_documents({"path": "/status"})))

    print("IPs:")
    ips = collection.aggregate([
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ])
    for ip in ips:
        print("\t{}: {}".format(ip.get("_id"), ip.get("count")))
