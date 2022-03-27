"""
model_handler.py

Provides support method to interact with the DQN model to produce results.
"""

from __future__ import print_function
from operator import itemgetter
import random
from dqn.main import generate_deferrable_appliance_data, generate_continuous_appliance_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# peak kWh charge ranges
off_peak = 0.082
mid_peak = 0.113
on_peak = 0.170


def generate_hour_grid(preferred, not_preferred):
    """
    Generates rated values that correspond with the user's preferences and
    optimal energy usage times used in the winter - https://www.oeb.ca/sites/default/files/tou-chart.pdf
    """

    # user convenience rates
    good_time = 0.1
    bad_time = 10

    energy_rates = {
        0: off_peak,
        1: off_peak,
        2: off_peak,
        3: off_peak,
        4: off_peak,
        5: off_peak,
        6: off_peak,
        7: on_peak,
        8: on_peak,
        9: on_peak,
        10: on_peak,
        11: mid_peak,
        12: mid_peak,
        13: mid_peak,
        14: mid_peak,
        15: mid_peak,
        16: mid_peak,
        17: on_peak,
        18: on_peak,
        19: off_peak,
        20: off_peak,
        21: off_peak,
        22: off_peak,
        23: off_peak,
    }

    for hour in range(0, 24):
        if hour in preferred:
            energy_rates[hour] = energy_rates[hour] * good_time
        elif hour in not_preferred:
            energy_rates[hour] = energy_rates[hour] * bad_time
    return energy_rates


def gausian_function(x, A, B):
    """
    Defines the Gaussian function.
    """

    y = A * np.exp(-1 * B * x**2)
    return y


def gaussian_fit(energy_rates):
    """
    Transforms the provided energy rates into a fitted gaussian function.
    """

    fitted_energy_hour_list = list(energy_rates.items())
    fitted_energy_hour_list = sorted(fitted_energy_hour_list, key=itemgetter(-1))
    sorted_hours = [x[0] for x in fitted_energy_hour_list]
    fitted_energy_hour_list = (
        fitted_energy_hour_list[::2] + fitted_energy_hour_list[-1::-2]
    )

    xdata = [
        -12.0,
        -11.0,
        -10.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
    ]
    ydata = [x[1] for x in fitted_energy_hour_list]
    hour_data = [x[0] for x in fitted_energy_hour_list]

    # Recast xdata and ydata into numpy arrays
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    parameters, covariance = curve_fit(gausian_function, xdata, ydata)

    fit_A = parameters[0]
    fit_B = parameters[1]

    fit_y = gausian_function(xdata, fit_A, fit_B)

    return fit_y, hour_data, sorted_hours


def generate_deferrable_historical_data(kilowatts, preferred):
    """
    Generates historical data for a given deferrable appliance based on the
    current user preferences.
    """

    ret = {}

    cost_rates = {
        0: off_peak,
        1: off_peak,
        2: off_peak,
        3: off_peak,
        4: off_peak,
        5: off_peak,
        6: off_peak,
        7: on_peak,
        8: on_peak,
        9: on_peak,
        10: on_peak,
        11: mid_peak,
        12: mid_peak,
        13: mid_peak,
        14: mid_peak,
        15: mid_peak,
        16: mid_peak,
        17: on_peak,
        18: on_peak,
        19: off_peak,
        20: off_peak,
        21: off_peak,
        22: off_peak,
        23: off_peak,
    }

    for day in range(0, 730):
        """
        Each day will store the total kilwatts used by that appliance,
        which hours of the day it was used, and the total cost based on the cost rates previously defined.
        """
        ret[day] = {"kw": 0, "hours": [], "cost": 0}

        # To account to user variance, how much an appliance gets used on a daily period and when is determined using guided-random data.
        random.shuffle(preferred)
        hours_used = random.randint(1, 21)
        if hours_used >= 16:
            hours_used = 2
        elif hours_used <= 2:
            hours_used = 0
        else:
            hours_used = 1

        for hour in preferred:
            if hours_used <= 0:
                break
            ret[day]["kw"] += kilowatts[0]
            ret[day]["hours"].append(hour)
            ret[day]["cost"] += kilowatts[0] * cost_rates[hour]
            hours_used -= 1

        ret[day]["cost"] = round(ret[day]["cost"], 2)

    return ret


def generate_deferrable_optimal_forecast_data(
    kilowatts, fit_y, hour_fit_alignment, optimal_hours
):
    """
    Uses the provided data to test the model, producing optimal forecasted data.
    """

    results = generate_deferrable_appliance_data(kilowatts, fit_y, hour_fit_alignment, optimal_hours)
    return results


def generate_continuous_historical_data(kilowatts):
    """
    Generates historical data for a given continuous appliance based on the
    current user preferences.
    """

    ret = {}

    cost_rates = {
        0: off_peak,
        1: off_peak,
        2: off_peak,
        3: off_peak,
        4: off_peak,
        5: off_peak,
        6: off_peak,
        7: on_peak,
        8: on_peak,
        9: on_peak,
        10: on_peak,
        11: mid_peak,
        12: mid_peak,
        13: mid_peak,
        14: mid_peak,
        15: mid_peak,
        16: mid_peak,
        17: on_peak,
        18: on_peak,
        19: off_peak,
        20: off_peak,
        21: off_peak,
        22: off_peak,
        23: off_peak,
    }

    for day in range(0, 730):
        """
        Each day will store the total kilwatts used by that appliance,
        which hours of the day it was used, and the total cost based on the cost rates previously defined.
        """

        ret[day] = {"kw": 0, "hours": [], "cost": 0}

        # To account to user variance, how much an appliance gets used on a daily period and when is determined using guided-random data.
        hours_list = [i for i in range(24)]
        random.shuffle(hours_list)
        hour_count = random.randint(12, 18)
        used_hours = []
        for h in range(hour_count):
            used_hours.append(hours_list[h])

        for hour in used_hours:
            ret[day]["kw"] += kilowatts[2]
            ret[day]["hours"].append(hour)
            ret[day]["cost"] += kilowatts[2] * cost_rates[hour]

        ret[day]["cost"] = round(ret[day]["cost"], 2)

    return ret


def generate_continuous_optimal_forecast_data(kilowatts):
    """
    Uses the provided data to test the model, producing optimal forecasted data.
    """

    results = generate_continuous_appliance_data(kilowatts)
    return results
