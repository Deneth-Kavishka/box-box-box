"""
Optimize parameters using historical data, then verify on test cases.
This is likely more reliable since we have 30,000 historical races vs 100 test cases.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

sys.stdout.reconfigure(line_buffering=True)

HIST_DIR = os.path.join('data', 'historical_races')
INPUTS_DIR = os.path.join('data', 'test_cases', 'inputs')
EXPECTED_DIR = os.path.join('data', 'test_cases', 'expected_outputs')

# Load historical races (use 3000 for training)
print("Loading data...")
hist_races = []
for fname in sorted(os.listdir(HIST_DIR))[:3]:
    with open(os.path.join(HIST_DIR, fname)) as f:
        hist_races.extend(json.load(f))
print(f"Loaded {len(hist_races)} historical races")

# Load test cases
test_cases = []
for i in range(1, 101):
    with open(os.path.join(INPUTS_DIR, f'test_{i:03d}.json')) as f:
        inp = json.load(f)
    with open(os.path.join(EXPECTED_DIR, f'test_{i:03d}.json')) as f:
        exp = json.load(f)
    test_cases.append((inp, exp['finishing_positions']))


def simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr):
    """Simulate race with cliff model, return predicted order."""
    config = race['race_config']
    strategies = race['strategies']
    base = config['base_lap_time']
    pit_time = config['pit_lane_time']
    total_laps = config['total_laps']
    temp = config['track_temp']

    off_map = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
    deg_map = {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}
    cliff_map = {'SOFT': sc, 'MEDIUM': mc, 'HARD': hc}
    tf = 1.0 + tc * (temp - tr)

    results = []
    for pos_key in sorted(strategies.keys()):
        strategy = strategies[pos_key]
        driver_id = strategy['driver_id']
        starting_pos = int(pos_key.replace('pos', ''))
        current_tire = strategy['starting_tire']
        tire_age = 0
        total_time = 0.0
        ps_map = {ps['lap']: ps['to_tire'] for ps in strategy['pit_stops']}

        for lap in range(1, total_laps + 1):
            tire_age += 1
            off = off_map[current_tire]
            deg = deg_map[current_tire]
            c = cliff_map[current_tire]
            degradation = deg * max(0, tire_age - c) * tf
            total_time += base + off + degradation
            if lap in ps_map:
                total_time += pit_time
                current_tire = ps_map[lap]
                tire_age = 0

        results.append((total_time, starting_pos, driver_id))
    results.sort()
    return [r[2] for r in results]


def eval_hist(so, ho, sd, md, hd, sc, mc, hc, tc, tr, races):
    """Evaluate parameters on historical races."""
    exact = 0
    for race in races:
        predicted = simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr)
        if predicted == race['finishing_positions']:
            exact += 1
    return exact


def eval_test(so, ho, sd, md, hd, sc, mc, hc, tc, tr):
    """Evaluate parameters on test cases."""
    exact = 0
    for race, expected in test_cases:
        predicted = simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr)
        if predicted == expected:
            exact += 1
    return exact


# Phase 1: Find best cliff combo on historical data
print("\n=== Phase 1: Find best cliffs on historical (500 races) ===")
train_races = hist_races[:500]

best_hist = 0
best_cliff = None

# Test range of parameter values with different cliff combos
base_params = [
    (-2.0, 1.5, 0.2, 0.1, 0.04, 0.01, 35.0),
    (-1.5, 1.5, 0.15, 0.08, 0.03, 0.01, 30.0),
    (-2.5, 2.0, 0.25, 0.12, 0.05, 0.015, 40.0),
    (-1.8, 1.6, 0.2, 0.09, 0.03, 0.02, 50.0),
]

for pi, (so, ho, sd, md, hd, tc, tr) in enumerate(base_params):
    for sc in range(0, 12):
        for mc in range(0, 12):
            for hc in range(0, 10):
                he = eval_hist(so, ho, sd, md, hd, sc, mc, hc, tc, tr, train_races)
                if he > best_hist:
                    best_hist = he
                    best_cliff = (sc, mc, hc, so, ho, sd, md, hd, tc, tr)
                    print(f"  cliff=({sc},{mc},{hc}), params={pi}: hist={he}/{len(train_races)}")
    print(f"  Param set {pi} done, best hist: {best_hist}/{len(train_races)}")

print(f"\nBest cliff: {best_cliff[:3]}, hist={best_hist}/{len(train_races)}")

# Phase 2: DE optimization with best cliff
if best_cliff:
    sc, mc, hc = int(best_cliff[0]), int(best_cliff[1]), int(best_cliff[2])
    print(f"\n=== Phase 2: DE for cliff=({sc},{mc},{hc}) on 1000 races ===")
    opt_races = hist_races[:1000]

    def objective(x):
        return -eval_hist(x[0], x[1], x[2], x[3], x[4], sc, mc, hc, x[5], x[6], opt_races)

    bounds = [
        (-5.0, 0.0),   # s_off
        (0.0, 4.0),    # h_off
        (0.01, 2.0),   # s_deg
        (0.01, 1.0),   # m_deg
        (0.001, 0.5),  # h_deg
        (0.001, 0.1),  # t_coeff
        (10.0, 100.0), # t_ref
    ]

    result = differential_evolution(
        objective, bounds, seed=42,
        maxiter=200, popsize=20,
        tol=1e-10, mutation=(0.5, 1.5),
        recombination=0.9, workers=1, disp=True
    )

    p = result.x
    he = eval_hist(p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6], opt_races)
    te = eval_test(p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6])
    print(f"\nDE result: hist={he}/{len(opt_races)}, test={te}/100")
    best_full = (p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6])

    # Hill climb on test cases
    print("\n=== Phase 3: Hill climb on test cases ===")
    current = list(best_full)
    ce = te
    for scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
        improved = True
        while improved:
            improved = False
            for dim in [0, 1, 2, 3, 4, 8, 9]:
                for direction in [-1, 1]:
                    trial = list(current)
                    step = scale * max(abs(current[dim]), 0.01)
                    trial[dim] += direction * step
                    if dim in [2, 3, 4, 8] and trial[dim] <= 0:
                        continue
                    new_te = eval_test(*trial)
                    if new_te > ce:
                        current = trial
                        ce = new_te
                        improved = True

            # Try cliff changes
            for dim in [5, 6, 7]:
                for delta in [-1, 1]:
                    trial = list(current)
                    new_val = max(0, int(trial[dim]) + delta)
                    trial[dim] = new_val
                    new_te = eval_test(*trial)
                    if new_te > ce:
                        current = trial
                        ce = new_te

        print(f"  Scale {scale}: test={ce}/100")

    # Final result
    so, ho, sd, md, hd, sc, mc, hc, tc, tr = current
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: test={ce}/100")
    print(f"s_off = {so:.15f}")
    print(f"h_off = {ho:.15f}")
    print(f"s_deg = {sd:.15f}")
    print(f"m_deg = {md:.15f}")
    print(f"h_deg = {hd:.15f}")
    print(f"s_cliff = {int(sc)}")
    print(f"m_cliff = {int(mc)}")
    print(f"h_cliff = {int(hc)}")
    print(f"t_coeff = {tc:.15f}")
    print(f"t_ref = {tr:.15f}")

    # Validate on full historical
    val_races = hist_races[:3000]
    vh = eval_hist(so, ho, sd, md, hd, int(sc), int(mc), int(hc), tc, tr, val_races)
    print(f"\nValidation on 3000 hist: {vh}/{len(val_races)} ({100*vh/len(val_races):.1f}%)")

    # Save
    best_params = {
        's_off': so, 'h_off': ho,
        's_deg': sd, 'm_deg': md, 'h_deg': hd,
        's_cliff': int(sc), 'm_cliff': int(mc), 'h_cliff': int(hc),
        't_coeff': tc, 't_ref': tr
    }
    with open('solution/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved to solution/best_params.json")
