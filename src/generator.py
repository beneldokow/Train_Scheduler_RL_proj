import random
import math

"""
RDDL Instance Generator for the Train Scheduler Project.

This module generates a valid RDDL instance file (.rddl) based on high-level parameters.
It calculates a "Perfect Schedule" (Ground Truth) based on a deterministic pass of the 
simulation physics, which the RL agent then tries to match in a stochastic environment.
"""
def generate_instance(
    num_trains,
    num_stations,
    horizon=None,
    interval=30,          # Time between train departures from the first station
    start_delay=10,       # Initial delay before the first train starts moving
    train_capacity=1000,
    boarding_speed=75,    # Passengers per minute
    variance_factor=0.2,  # Scales the variance of passenger arrivals
):
    """
    Generates a complete RDDL instance string.
    
    Logic Flow:
    1. Define static network topology (stations, drive times).
    2. Calculate passenger arrival rates and disembarking percentages per station.
    3. Run a deterministic simulation to find the 'Perfect Departure Time' for every (train, station) pair.
    4. Format and return the RDDL non-fluents and instance definition.
    """

    if num_trains < 2 or num_stations < 2:
        raise ValueError("There must be at least 2 trains and 2 stations.")

    trains = [f"t{i}" for i in range(1, num_trains + 1)]
    stations = [f"s{i}" for i in range(1, num_stations + 1)]

    # 1. Network Topology: Random but fixed drive times between stations
    drive_times = {
        (stations[i], stations[i + 1]): random.randint(10, 60)
        for i in range(len(stations) - 1)
    }

    # 2. Initial State: Staggered starts for trains at the first station
    train_timers = [start_delay + (i * interval) for i in range(num_trains)]

    # 3. Passenger Flow Parameters
    station_weights = [random.randint(1, 10) for _ in range(num_stations)]
    total_weight = sum(station_weights)
    if total_weight == 0: total_weight = 1

    disembarking_percentages = {
        stations[i]: station_weights[i] / total_weight for i in range(num_stations)
    }

    passenger_arrival_rates = {}
    passenger_arrival_vars = {}
    for i, station in enumerate(stations):
        if i == len(stations) - 1:  # The Depot (final stop) has no new arrivals
            passenger_arrival_rates[station] = 0.0
            passenger_arrival_vars[station] = 0.0
        else:
            rate = random.randint(2, 8)
            passenger_arrival_rates[station] = float(rate)
            # Variance is derived from the mean to keep stochasticity physically plausible
            passenger_arrival_vars[station] = float(rate) * variance_factor

    # 4. Schedule Calculation (Ground Truth)
    # We simulate a "perfect world" where there is no stochastic variance and no agent intervention.
    # The resulting 'planned_departures' are the target labels for the RL agent.
    planned_departures = {}
    station_free_time = {s: 0.0 for s in stations} # Tracks when a station's platform becomes empty
    station_last_snapshot_time = {s: 0.0 for s in stations} # Tracks last time passengers were picked up
    station_leftover_passengers = {s: 0.0 for s in stations}

    for t_idx, train in enumerate(trains):
        current_train_load = 0.0
        natural_arrival_time = train_timers[t_idx]

        for s_idx, station in enumerate(stations):
            # Calculate drive time from previous station
            if s_idx > 0:
                prev_station = stations[s_idx - 1]
                drive = drive_times[(prev_station, station)]
                natural_arrival_time += drive

            # Accumulation Logic: How many passengers have arrived since the LAST train left?
            accumulation_interval = natural_arrival_time - station_last_snapshot_time[station]
            if accumulation_interval < 0: accumulation_interval = 0

            new_arrivals = accumulation_interval * passenger_arrival_rates[station]
            total_waiting = new_arrivals + station_leftover_passengers[station]
            station_last_snapshot_time[station] = natural_arrival_time

            # Blocking Logic: A train cannot board until the platform is clear
            actual_boarding_start = max(natural_arrival_time, station_free_time[station])

            # Dwell Time Logic: Time spent boarding and disembarking
            want_to_get_off = current_train_load * disembarking_percentages[station]
            current_train_load -= want_to_get_off
            
            space_available = train_capacity - current_train_load
            actually_boarding = min(total_waiting, space_available)
            current_train_load += actually_boarding
            station_leftover_passengers[station] = total_waiting - actually_boarding

            # Dwell time is the bottleneck between boarding and disembarking speed
            dwell_time = math.ceil(max(want_to_get_off, actually_boarding) / boarding_speed)

            # Record the 'Perfect' Departure Time
            departure_time = actual_boarding_start + dwell_time
            planned_departures[(train, station)] = departure_time

            # Update station and train state for the next calculation
            station_free_time[station] = departure_time
            natural_arrival_time = departure_time

    # Auto-Horizon: Ensure the simulation is long enough for all trains to finish
    if horizon is None:
        horizon = (2 * num_trains * num_stations) + 10

    # 5. RDDL Formatting
    output = []
    output.append("non-fluents nf_simple_train_model{\n")
    output.append("    domain = train_system;\n\n")

    output.append("    objects{\n")
    output.append(f"        train : {{{', '.join(trains)}}};\n")
    output.append(f"        station : {{{', '.join(stations)}}};\n")
    output.append("    };\n\n")

    output.append("    non-fluents{\n")

    # Define train ordering for queue management
    for i in range(num_trains):
        next_train = trains[(i + 1) % num_trains]
        output.append(f"        NEXT_TRAIN({trains[i]}, {next_train}) = true;\n")

    output.append(f"\n        DEPOT_STATION({stations[-1]}) = true;\n\n")

    # Network connections
    for i in range(len(stations) - 1):
        output.append(f"        NEXT_STATION({stations[i]}, {stations[i+1]}) = true;\n")
        output.append(f"        FIND_NEXT_STATION({stations[i]}) = {stations[i+1]};\n")
    output.append(f"        FIND_NEXT_STATION({stations[-1]}) = {stations[-1]};\n")

    output.append(f"        FIRST_TRAIN = {trains[0]};\n\n")

    # Drive times
    for (s1, s2), time in drive_times.items():
        output.append(f"        DRIVE_TIME({s1}, {s2}) = {time};\n")

    # Passenger behavior
    output.append("\n")
    for station, percentage in disembarking_percentages.items():
        output.append(f"        DISEMBARKING_PRECENTAGE({station}) = {percentage:.2f};\n")

    output.append("\n")
    for station, rate in passenger_arrival_rates.items():
        output.append(f"        PASSENGER_ARRIVAL_RATE({station}) = {rate:.1f};\n")
        output.append(f"        PASSENGER_ARRIVAL_VAR({station}) = {passenger_arrival_vars[station]:.1f};\n")

    # Target Schedule
    output.append("\n")
    for (train, station), time in planned_departures.items():
        output.append(f"        PLANNED_DEPARTURE_TIME({train}, {station}) = {int(time)};\n")

    output.append("\n    };\n}\n\n")

    # Instance definition (Initial State)
    output.append("instance simple_train_model{\n")
    output.append("    domain = train_system;\n")
    output.append("    non-fluents = nf_simple_train_model;\n\n")

    output.append("    init-state{\n")
    for i, train in enumerate(trains):
        output.append(f"        train_timer({train}) = {train_timers[i]};\n")
        output.append(f"        current_state({train}) = 0;\n")
        output.append(f"        current_station({train}) = {stations[0]};\n")
    output.append("    };\n\n")

    output.append(f"    max-nondef-actions = pos-inf;\n")
    output.append(f"    horizon = {horizon};\n")
    output.append("    discount = 1.0;\n\n")
    output.append("}\n")

    return "".join(output)
