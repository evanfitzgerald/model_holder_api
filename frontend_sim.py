import sys
import json
import requests

'''
full_conv = [
            {
                'washer': {'preferred': [6, 7, 16, 17, 18, 19], 'not_preferred': [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 21, 22, 23]}, 
                'dryer': {'preferred': [], 'not_preferred': []},
                'dishwasher': {'preferred': [], 'not_preferred': []}, 
                'HVAC': {'preferred': [], 'not_preferred': []},
            }
        ]
'''
conv = [
            {'washer': {'preferred': [6, 7, 16, 17, 18, 19], 'not_preferred': [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 21, 22, 23], 'kw': 0.5}}, 
            {'dryer': {'preferred': [7, 8, 17, 18, 19, 20], 'not_preferred': [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 22, 23], 'kw': 2.25}},
            {'dishwasher': {'preferred': [12, 13, 18, 19], 'not_preferred': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 21, 22, 23], 'kw': 1.25}},
            {'HVAC': {'preferred': [], 'not_preferred': [], 'kw': [0, 0.75, 3]}}
        ]
s = json.dumps(conv)
res = requests.post("http://127.0.0.1:5000/", json=s).json()

result = json.dumps(res, indent = 4)
        
with open("output.json", "w") as outfile:
    outfile.write(result)