from flask import Flask
from flask import request
import json
from operator import itemgetter

from model_handler import auto_generate_historical_data, auto_generate_optimal_forecast_data, generate_historical_data, generate_hour_grid, gaussian_fit, generate_optimal_forecast_data

app = Flask(__name__) 

@app.route('/', methods = ['POST'])
def generate_model_results():
    jsondata = request.get_json()
    data = json.loads(jsondata)

    #stuff happens here that involves data to obtain a result
    #print(data)

    output_data = {}

    for appliance in data:
        appliance_name = list(appliance.keys())[0]
        preferred = appliance[appliance_name]['preferred']
        not_preferred = appliance[appliance_name]['not_preferred']
        kilowatts = appliance[appliance_name]['kw']

        if preferred != [] and not_preferred != []:
            energy_rates = generate_hour_grid(preferred, not_preferred)
            fit_y, hour_fit_alignment, optimal_hours = gaussian_fit(energy_rates)

            historical_results = generate_historical_data(kilowatts, preferred)
            forecast_results = generate_optimal_forecast_data(kilowatts, fit_y, hour_fit_alignment, optimal_hours)

        else: # automatic system (fridge, hvac, etc.), includes list of kilowatts
            historical_results = auto_generate_historical_data(kilowatts)
            forecast_results = auto_generate_optimal_forecast_data(kilowatts)

        output_data[appliance_name + '_historical'] = historical_results
        output_data[appliance_name + '_forecast'] = forecast_results
        
        '''
        historical_results_json = json.dumps(historical_results, indent = 4)
        forecast_results_json = json.dumps(forecast_results, indent = 4)
        
        with open("historical_results_" + appliance_name + ".json", "w") as outfile:
            outfile.write(historical_results_json)

        with open("forecast_results_" + appliance_name + ".json", "w") as outfile:
            outfile.write(forecast_results_json)
        '''

    return json.dumps(output_data)

if __name__ == '__main__':
    app.run(debug=True)