#!/usr/bin/env python3
"""
Box Box Box - F1 Race Simulator
Predicts finishing positions based on tire strategies and race conditions.
"""

import json
import sys


def simulate_race(test_case):
    """
    Simulate race lap-by-lap and return finishing positions.

    The simulation uses a cliff-based tire degradation model:
    - Each tire compound has an offset (speed characteristic)
    - Tire performance degrades after a "cliff" period
    - Temperature affects degradation rate
    """
    config = test_case['race_config']
    strategies = test_case['strategies']

    base_lap_time = config['base_lap_time']
    pit_lane_time = config['pit_lane_time']
    total_laps = config['total_laps']
    track_temp = config['track_temp']

    # Model parameters (optimized from historical data)
    COMPOUND_OFFSET = {'SOFT': -1.78454790726348, 'MEDIUM': 0.0, 'HARD': 1.57724006309548}
    DEG_RATE = {'SOFT': 1.10395810062795, 'MEDIUM': 0.27776420033883, 'HARD': 0.01890892407769}
    CLIFF = {'SOFT': 2, 'MEDIUM': 7, 'HARD': 3}
    TEMP_COEFF = 0.02578582369718
    TEMP_REF = 74.33919475487491

    # Temperature factor affects degradation
    temp_factor = 1.0 + TEMP_COEFF * (track_temp - TEMP_REF)

    driver_times = []

    for pos_key in strategies:
        strategy = strategies[pos_key]
        driver_id = strategy['driver_id']
        starting_pos = int(pos_key.replace('pos', ''))
        current_tire = strategy['starting_tire']
        tire_age = 0
        total_time = 0.0

        # Create pit stop lookup
        pit_stops = {ps['lap']: ps['to_tire'] for ps in strategy['pit_stops']}

        # Simulate each lap
        for lap in range(1, total_laps + 1):
            # Tire age increments at start of lap
            tire_age += 1

            # Calculate lap time components
            offset = COMPOUND_OFFSET[current_tire]
            deg = DEG_RATE[current_tire]
            cliff = CLIFF[current_tire]

            # Degradation only applies after cliff period
            degradation = deg * max(0, tire_age - cliff) * temp_factor

            lap_time = base_lap_time + offset + degradation
            total_time += lap_time

            # Handle pit stop at end of lap
            if lap in pit_stops:
                total_time += pit_lane_time
                current_tire = pit_stops[lap]
                tire_age = 0

        driver_times.append((driver_id, total_time, starting_pos))

    # Sort by total time, using starting position as tiebreaker
    driver_times.sort(key=lambda x: (x[1], x[2]))
    return [d[0] for d in driver_times]


def main():
    """Read input from stdin and output prediction to stdout."""
    test_case = json.load(sys.stdin)
    finishing_positions = simulate_race(test_case)

    output = {
        'race_id': test_case['race_id'],
        'finishing_positions': finishing_positions
    }

    print(json.dumps(output))


if __name__ == '__main__':
    main()
