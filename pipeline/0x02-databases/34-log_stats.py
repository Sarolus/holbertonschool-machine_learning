#!/usr/bin/env python3

from pymongo import MongoClient
import urllib

if __name__ == "__main__":
    username = urllib.parse.quote_plus('root')
    password = urllib.parse.quote_plus('root')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    database = client.logs
    collection = database.nginx

    print("{} logs".format(collection.count_documents({})))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        print("\tmethod {}: {}".format(method, collection.count_documents({"method": method})))

    print("{} status check".format(collection.count_documents({"path": "/status"})))
