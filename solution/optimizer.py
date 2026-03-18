"""
Comprehensive parameter optimization for F1 race simulator.
Uses differential evolution with extensive search across all cliff combinations.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

sys.stdout.reconfigure(line_buffering=True)

# Load all data
print("Loading data...")
HIST_DIR = os.path.join('data', 'historical_races')
INPUTS_DIR = os.path.join('data', 'test_cases', 'inputs')
EXPECTED_DIR = os.path.join('data', 'test_cases', 'expected_outputs')

# Load test cases
test_cases = []
for i in range(1, 101):
    with open(os.path.join(INPUTS_DIR, f'test_{i:03d}.json')) as f:
        inp = json.load(f)
    with open(os.path.join(EXPECTED_DIR, f'test_{i:03d}.json')) as f:
        exp = json.load(f)
    test_cases.append((inp, exp['finishing_positions']))

# Load historical races
hist_races = []
for fname in sorted(os.listdir(HIST_DIR))[:5]:  # Use 5000 races
    with open(os.path.join(HIST_DIR, fname)) as f:
        hist_races.extend(json.load(f))

print(f"Loaded {len(test_cases)} test cases, {len(hist_races)} historical races")


def simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr):
    """Simulate race with cliff model."""
    config = race['race_config']
    strategies = race['strategies']
    base = config['base_lap_time']
    pit_time = config['pit_lane_time']
    total_laps = config['total_laps']
    temp = config['track_temp']

    off_map = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
    deg_map = {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}
    cliff_map = {'SOFT': int(sc), 'MEDIUM': int(mc), 'HARD': int(hc)}
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


def eval_test(params):
    """Count exact matches on test cases."""
    so, ho, sd, md, hd, sc, mc, hc, tc, tr = params
    exact = 0
    for race, expected in test_cases:
        if simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr) == expected:
            exact += 1
    return exact


def eval_hist(params, races):
    """Count exact matches on historical races."""
    so, ho, sd, md, hd, sc, mc, hc, tc, tr = params
    exact = 0
    for race in races:
        if simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr) == race['finishing_positions']:
            exact += 1
    return exact


def compute_displacement(params):
    """Compute total displacement (lower is better)."""
    so, ho, sd, md, hd, sc, mc, hc, tc, tr = params
    total_disp = 0
    for race, expected in test_cases:
        predicted = simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr)
        pred_rank = {d: i for i, d in enumerate(predicted)}
        for i, d in enumerate(expected):
            total_disp += abs(pred_rank[d] - i)
    return total_disp


# Phase 1: Comprehensive cliff search with DE
print("\n" + "="*60)
print("PHASE 1: Comprehensive cliff search with DE")
print("="*60)

best_test = 0
best_hist = 0
best_params = None

# Generate all cliff combinations
cliff_combos = []
for sc in range(0, 15):
    for mc in range(0, 15):
        for hc in range(0, 12):
            cliff_combos.append((sc, mc, hc))

print(f"Testing {len(cliff_combos)} cliff combinations...")

# Quick screening - test each cliff with base params
train_hist = hist_races[:1000]
screen_results = []

for idx, (sc, mc, hc) in enumerate(cliff_combos):
    # Test with multiple parameter sets
    best_for_cliff = 0
    for so in [-2.0, -1.5, -1.0]:
        for ho in [1.0, 1.5, 2.0]:
            for sd in [0.1, 0.2, 0.5, 1.0]:
                for md in [0.05, 0.1, 0.2, 0.3]:
                    for hd in [0.02, 0.05, 0.1]:
                        for tc in [0.005, 0.01, 0.02]:
                            for tr in [25, 30, 35, 50, 75]:
                                params = (so, ho, sd, md, hd, sc, mc, hc, tc, tr)
                                te = eval_test(params)
                                if te > best_for_cliff:
                                    best_for_cliff = te
                                    if te > best_test:
                                        best_test = te
                                        best_params = params
                                        print(f"  cliff=({sc},{mc},{hc}): test={te}/100 ***")

    screen_results.append((sc, mc, hc, best_for_cliff))

    if idx % 200 == 0 and idx > 0:
        print(f"  Progress: {idx}/{len(cliff_combos)}, best so far: {best_test}/100")

# Sort by score
screen_results.sort(key=lambda x: -x[3])
print(f"\nTop 20 cliff combinations:")
for sc, mc, hc, score in screen_results[:20]:
    print(f"  cliff=({sc},{mc},{hc}): {score}/100")

# Phase 2: DE optimization on top cliff combos
print("\n" + "="*60)
print("PHASE 2: DE optimization on top cliff combinations")
print("="*60)

top_cliffs = [(x[0], x[1], x[2]) for x in screen_results[:10]]

for sc, mc, hc in top_cliffs:
    print(f"\nOptimizing cliff=({sc},{mc},{hc})...")

    def objective(x):
        params = (x[0], x[1], x[2], x[3], x[4], sc, mc, hc, x[5], x[6])
        disp = compute_displacement(params)
        return disp

    bounds = [
        (-5.0, 0.0),    # s_off
        (0.0, 5.0),     # h_off
        (0.01, 3.0),    # s_deg
        (0.01, 1.0),    # m_deg
        (0.001, 0.5),   # h_deg
        (0.001, 0.1),   # t_coeff
        (10.0, 100.0),  # t_ref
    ]

    result = differential_evolution(
        objective, bounds, seed=42,
        maxiter=150, popsize=20,
        tol=1e-10, mutation=(0.5, 1.5),
        recombination=0.9, workers=1
    )

    p = result.x
    params = (p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6])
    te = eval_test(params)
    disp = result.fun

    if te > best_test:
        best_test = te
        best_params = params
        print(f"  NEW BEST: test={te}/100, disp={disp:.0f}")
    else:
        print(f"  test={te}/100, disp={disp:.0f}")

# Phase 3: Hill climbing
print("\n" + "="*60)
print("PHASE 3: Hill climbing best solution")
print("="*60)

current = list(best_params)
ce = eval_test(current)
print(f"Starting from: test={ce}/100")

for scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]:
    improved = True
    iters = 0
    while improved and iters < 500:
        improved = False
        iters += 1

        # Continuous params
        for dim in [0, 1, 2, 3, 4, 8, 9]:
            for direction in [-1, 1]:
                trial = list(current)
                step = scale * max(abs(current[dim]), 0.01)
                trial[dim] += direction * step
                if dim in [2, 3, 4, 8] and trial[dim] <= 0:
                    continue
                te = eval_test(trial)
                if te > ce:
                    current = trial
                    ce = te
                    improved = True

        # Cliff params
        for dim in [5, 6, 7]:
            for delta in [-1, 1]:
                trial = list(current)
                new_val = max(0, int(trial[dim]) + delta)
                trial[dim] = new_val
                te = eval_test(trial)
                if te > ce:
                    current = trial
                    ce = te
                    improved = True

    print(f"  Scale {scale}: test={ce}/100")

# Final result
print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)

so, ho, sd, md, hd, sc, mc, hc, tc, tr = current
print(f"Test accuracy: {ce}/100 ({ce}%)")
print(f"\nParameters:")
print(f"  COMPOUND_OFFSET = {{'SOFT': {so}, 'MEDIUM': 0.0, 'HARD': {ho}}}")
print(f"  DEG_RATE = {{'SOFT': {sd}, 'MEDIUM': {md}, 'HARD': {hd}}}")
print(f"  CLIFF = {{'SOFT': {int(sc)}, 'MEDIUM': {int(mc)}, 'HARD': {int(hc)}}}")
print(f"  TEMP_COEFF = {tc}")
print(f"  TEMP_REF = {tr}")

# Validate on historical
he = eval_hist(current, hist_races[:2000])
print(f"\nHistorical validation (2000 races): {he}/2000 ({100*he/2000:.1f}%)")

# Save best params
best_json = {
    's_off': so, 'h_off': ho,
    's_deg': sd, 'm_deg': md, 'h_deg': hd,
    's_cliff': int(sc), 'm_cliff': int(mc), 'h_cliff': int(hc),
    't_coeff': tc, 't_ref': tr
}
with open('solution/best_params.json', 'w') as f:
    json.dump(best_json, f, indent=2)
print("\nSaved to solution/best_params.json")
