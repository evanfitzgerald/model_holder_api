
from __future__ import print_function
from operator import itemgetter
import random
from dqn.main import get_train_results, get_auto_train_results
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def generate_hour_grid(preferred, not_preferred):
    # time convenience
    good_time = 0.1
    bad_time = 10

    # peak kWh charge ranges 
    off_peak = 0.082
    mid_peak = 0.113
    on_peak = 0.170

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

def Gauss(x, A, B):
    # Define the Gaussian function
    y = A*np.exp(-1*B*x**2)
    return y

def gaussian_fit(energy_rates):
    fitted_energy_hour_list = list(energy_rates.items())
    fitted_energy_hour_list = sorted(fitted_energy_hour_list, key=itemgetter(-1))
    sorted_hours = [x[0] for x in fitted_energy_hour_list]
    fitted_energy_hour_list = fitted_energy_hour_list[::2] + fitted_energy_hour_list[-1::-2]

    xdata = [ -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    ydata = [x[1] for x in fitted_energy_hour_list]
    
    hour_data = [x[0] for x in fitted_energy_hour_list]

    # Recast xdata and ydata into numpy arrays
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
    parameters, covariance = curve_fit(Gauss, xdata, ydata)
    
    fit_A = parameters[0]
    fit_B = parameters[1]
    
    fit_y = Gauss(xdata, fit_A, fit_B)

    return fit_y, hour_data, sorted_hours

def generate_historical_data(kilowatts, preferred):
    json_obj = {}

    # peak kWh charge ranges 
    off_peak = 0.082
    mid_peak = 0.113
    on_peak = 0.170

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
        json_obj[day] = {'kw': 0, 'hours': [], 'cost': 0}
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
            json_obj[day]['kw'] += kilowatts
            json_obj[day]['hours'].append(hour)
            json_obj[day]['cost'] += kilowatts * cost_rates[hour]
            hours_used -= 1

        json_obj[day]['cost'] = round(json_obj[day]['cost'], 2)

    return json_obj

def generate_optimal_forecast_data(kilowatts, fit_y, hour_fit_alignment, optimal_hours):
    results = get_train_results(kilowatts, fit_y, hour_fit_alignment, optimal_hours)
    return results

def auto_generate_historical_data(kilowatts):
    json_obj = {}

    # peak kWh charge ranges 
    off_peak = 0.082
    mid_peak = 0.113
    on_peak = 0.170

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
        json_obj[day] = {'kw': 0, 'hours': [], 'cost': 0}
        hours_list = [i for i in range(24)]
        random.shuffle(hours_list)
        hour_count = random.randint(12, 18)
        used_hours = []
        for h in range(hour_count):
            used_hours.append(hours_list[h])

        for hour in used_hours:
            json_obj[day]['kw'] += kilowatts[2]
            json_obj[day]['hours'].append(hour)
            json_obj[day]['cost'] += kilowatts[2] * cost_rates[hour]

        json_obj[day]['cost'] = round(json_obj[day]['cost'], 2)

    return json_obj

def auto_generate_optimal_forecast_data(kilowatts):
    results = get_auto_train_results(kilowatts)
    return results


