#!/usr/bin/env python3
"""
    API Requesting Module
"""

import requests


def availableShips(passengerCount):
    """
        Method that returns the list of ships that can hold
        a given number of passengers.

        Args:
            passengerCount (int): The given number of passengers.

        Returns:
            ship_list (list): The list of the ships.
    """

    ship_list = []
    swapi_url = "https://swapi-api.hbtn.io/api/starships"

    while swapi_url is not None:
        data = requests.get(swapi_url).json()
        for ship in data["results"]:
            passengers = ship["passengers"]
            if passengers != "n/a" and passengers != "unknown":
                if int(passengers.replace(",", "")) >= passengerCount:
                    ship_list.append(ship["name"])
        swapi_url = data["next"]

    return ship_list
