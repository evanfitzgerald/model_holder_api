"""
tester.py

Acts as a requester to the model resource for testing, ensures feedback is accurate.
"""

import sys
import json
from unicodedata import name
import requests

test_appliances = [
    {
        "washer": {
            "preferred": [6, 7, 16, 17, 18, 19],
            "not_preferred": [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 21, 22, 23],
            "kw": [0.5],
        }
    },
    {
        "dryer": {
            "preferred": [7, 8, 17, 18, 19, 20],
            "not_preferred": [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 22, 23],
            "kw": [2.25],
        }
    },
    {
        "dishwasher": {
            "preferred": [12, 13, 18, 19],
            "not_preferred": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 21, 22, 23],
            "kw": [1.25],
        }
    },
    {"HVAC": {"preferred": [], "not_preferred": [], "kw": [0, 0.75, 3]}},
]


if __name__ ==  '__main__':
    url_address = "http://127.0.0.1:5000/"
    appliance_json = json.dumps(test_appliances)
    response = requests.post(url_address, json=appliance_json).json()
    result = json.dumps(response, indent=4)

    with open("tester_feedback.json", "w") as outfile:
        outfile.write(result)
