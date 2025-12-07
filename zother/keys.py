__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json

PATH = "C:/Users/Dylan/projects/keys.json"
with open(PATH, 'r') as file:
    KEYS = json.load(file)


def get_api_key(service: str) -> str:
    """
    Retrieve the API key for a given service from the keys.json file.
    :param service: Name of the service to retrieve the API key for
    :return: API key as a string
    """
    # Raise an error immediately if the key is not found
    key = KEYS.get(service)
    if key is None:
        raise ValueError(f"API key for service '{service}' not found.")
    return key
