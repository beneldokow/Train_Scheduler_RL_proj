import random
import math


def generate_instance(
    num_trains,
    num_stations,
    horizon,
    interval=30,
    start_delay=10,
    train_capacity=1000,
    boarding_speed=75,
):

    if num_trains < 2 or num_stations < 2:
        raise ValueError("There must be at least 2 trains and 2 stations.")

    trains = [f"t{i}" for i in range(1, num_trains + 1)]
    stations = [f"s{i}" for i in range(1, num_stations + 1)]

    # Define random drive times
    drive_times = {
        (stations[i], stations[i + 1]): random.randint(10, 60)
        for i in range(len(stations) - 1)
    }

    # t1 starts at 'start_delay', t2 at 'start_delay + interval', etc.
    # This allows passengers to accumulate at s1 before t1 arrives.
    train_timers = [start_delay + (i * interval) for i in range(num_trains)]

    # Disembarking Percentages
    station_weights = [random.randint(1, 10) for _ in range(num_stations)]
    total_weight = sum(station_weights)
    if total_weight == 0:
        total_weight = 1

    disembarking_percentages = {
        stations[i]: station_weights[i] / total_weight for i in range(num_stations)
    }

    # Passenger Arrival Rates
    # Last station (Depot) gets 0 arrival rate. s1 gets a normal random rate.
    passenger_arrival_rates = {}
    for i, station in enumerate(stations):
        if i == len(stations) - 1:  # Last station
            passenger_arrival_rates[station] = 0
        else:
            passenger_arrival_rates[station] = random.randint(2, 8)

    # --- SIMULATION START ---
    planned_departures = {}

    station_free_time = {s: 0.0 for s in stations}
    station_last_snapshot_time = {s: 0.0 for s in stations}
    station_leftover_passengers = {s: 0.0 for s in stations}

    for t_idx, train in enumerate(trains):
        current_train_load = 0.0

        # 1. Determine Natural Arrival at s1 (includes the start_delay now)
        natural_arrival_time = train_timers[t_idx]

        for s_idx, station in enumerate(stations):
            if s_idx > 0:
                prev_station = stations[s_idx - 1]
                drive = drive_times[(prev_station, station)]
                natural_arrival_time += drive

            # --- SNAPSHOT LOGIC ---
            # For t1 at s1, accumulation_interval will be 'start_delay' (e.g. 10 - 0 = 10 mins).
            accumulation_interval = (
                natural_arrival_time - station_last_snapshot_time[station]
            )
            if accumulation_interval < 0:
                accumulation_interval = 0

            new_arrivals = accumulation_interval * passenger_arrival_rates[station]
            total_waiting = new_arrivals + station_leftover_passengers[station]

            station_last_snapshot_time[station] = natural_arrival_time

            # --- BLOCKING LOGIC ---
            actual_boarding_start = max(
                natural_arrival_time, station_free_time[station]
            )

            # --- CAPACITY & DWELL ---
            # Disembark
            want_to_get_off = current_train_load * disembarking_percentages[station]
            current_train_load -= want_to_get_off
            if current_train_load < 0:
                current_train_load = 0

            # Board
            space_available = train_capacity - current_train_load

            if total_waiting <= space_available:
                actually_boarding = total_waiting
                leftover = 0
            else:
                actually_boarding = space_available
                leftover = total_waiting - space_available

            current_train_load += actually_boarding
            station_leftover_passengers[station] = leftover

            # Dwell Time
            time_to_disembark = want_to_get_off / boarding_speed
            time_to_board = actually_boarding / boarding_speed

            raw_dwell = max(time_to_disembark, time_to_board)
            dwell_time = math.ceil(raw_dwell)

            # --- DEPARTURE ---
            departure_time = actual_boarding_start + dwell_time
            planned_departures[(train, station)] = departure_time

            station_free_time[station] = departure_time
            natural_arrival_time = departure_time

    # --- OUTPUT GENERATION ---
    output = []
    output.append("non-fluents nf_simple_train_model{\n")
    output.append("    domain = train_system;\n\n")

    output.append("    objects{\n")
    output.append(f"        train : {{{', '.join(trains)}}};\n")
    output.append(f"        station : {{{', '.join(stations)}}};\n")
    output.append("    };\n\n")

    output.append("    non-fluents{\n")

    for i in range(num_trains):
        next_train = trains[(i + 1) % num_trains]
        output.append(f"        NEXT_TRAIN({trains[i]}, {next_train}) = true;\n")

    output.append(f"\n        DEPOT_STATION({stations[-1]}) = true;\n\n")

    for i in range(len(stations) - 1):
        output.append(f"        NEXT_STATION({stations[i]}, {stations[i+1]}) = true;\n")
        output.append(f"        FIND_NEXT_STATION({stations[i]}) = {stations[i+1]};\n")

    output.append(f"        FIND_NEXT_STATION({stations[-1]}) = {stations[-1]};\n")

    output.append(f"        FIRST_TRAIN = {trains[0]};\n")

    output.append("\n")
    for (s1, s2), time in drive_times.items():
        output.append(f"        DRIVE_TIME({s1}, {s2}) = {time};\n")

    output.append("\n")
    for station, percentage in disembarking_percentages.items():
        output.append(
            f"        DISEMBARKING_PRECENTAGE({station}) = {percentage:.2f};\n"
        )

    output.append("\n")
    for station, rate in passenger_arrival_rates.items():
        output.append(f"        PASSENGER_ARRIVAL_RATE({station}) = {rate};\n")

    output.append("\n")
    for (train, station), time in planned_departures.items():
        output.append(
            f"        PLANNED_DEPARTURE_TIME({train}, {station}) = {int(time)};\n"
        )

    output.append("\n    };\n")
    output.append("}\n\n")

    output.append("instance simple_train_model{\n")
    output.append("    domain = train_system;\n")
    output.append("    non-fluents = nf_simple_train_model;\n\n")

    output.append("    init-state{\n")
    for i, train in enumerate(trains):
        output.append(f"        train_timer({train}) = {train_timers[i]};\n")

    output.append("\n")
    for train in trains:
        output.append(f"        current_state({train}) = {0};\n")
        output.append(f"        current_station({train}) = {stations[0]};\n")

    output.append("    };\n\n")
    output.append(f"    max-nondef-actions = pos-inf;\n")
    output.append(f"    horizon = {horizon};\n")
    output.append("    discount = 1.0;\n\n")
    output.append("}\n")

    return "".join(output)


pass
