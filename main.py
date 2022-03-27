"""
main.py

Initializes Flask API to allow the app to interact with the model.
"""

from flask import Flask, request
import json
from operator import itemgetter

from model_handler import (
    generate_continuous_historical_data,
    generate_continuous_optimal_forecast_data,
    generate_deferrable_historical_data,
    generate_hour_grid,
    gaussian_fit,
    generate_deferrable_optimal_forecast_data,
)

app = Flask(__name__)


def is_deferrable_load(kilowatts):
    return len(kilowatts) == 1


@app.route("/", methods=["POST"])
def generate_model_results():
    """
    For each appliance provided by the user, two years of historical
    and forecasted data is generated and returned.
    """

    ret = {}

    # Collect incoming appliance data
    json_data = request.get_json()
    appliance_data = json.loads(json_data)

    for appliance in appliance_data:
        appliance_name = list(appliance.keys())[0]
        preferred_hours = appliance[appliance_name]["preferred"]
        not_preferred_hours = appliance[appliance_name]["not_preferred"]
        kilowatts = appliance[appliance_name]["kw"]

        # not_preferred_hours dictates whether the appliance is a deferrable or continuous load
        if is_deferrable_load(kilowatts):
            energy_rates = generate_hour_grid(preferred_hours, not_preferred_hours)
            fit_y, hour_fit_alignment, optimal_hours = gaussian_fit(energy_rates)

            historical_results = generate_deferrable_historical_data(
                kilowatts, preferred_hours
            )
            forecast_results = generate_deferrable_optimal_forecast_data(
                kilowatts, fit_y, hour_fit_alignment, optimal_hours
            )

        else:  # continuous load, e.g. HVAC, uses a list of kilowatts
            historical_results = generate_continuous_historical_data(kilowatts)
            forecast_results = generate_continuous_optimal_forecast_data(kilowatts)

        ret[appliance_name + "_historical"] = historical_results
        ret[appliance_name + "_forecast"] = forecast_results

    return json.dumps(ret)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
