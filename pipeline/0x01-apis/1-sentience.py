#!/usr/bin/env python3
"""
    API Requesting Module
"""

import requests


def sentientPlanets():
    """
        Method that returns the list of names of the
        home planets of all sentient species.
    """

    planet_list = []
    swapi_url = 'https://swapi-api.hbtn.io/api/species'

    while swapi_url is not None:
        data = requests.get(swapi_url).json()
        for species in data['results']:
            if ((species['designation'] == 'sentient'
                 or species['designation'] == 'reptilian')):
                if species['homeworld'] is not None:
                    homeworld = requests.get(species['homeworld']).json()
                    planet_list.append(homeworld['name'])
        swapi_url = data['next']

    return planet_list
