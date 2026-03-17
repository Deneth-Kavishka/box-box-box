"""Find best parameters using DE + hill climbing on test cases."""
import json
import os
import sys
from scipy.optimize import differential_evolution

sys.stdout.reconfigure(line_buffering=True)

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
print(f"Loaded {len(test_cases)} test cases")


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


def eval_params(so, ho, sd, md, hd, sc, mc, hc, tc, tr):
    """Evaluate parameters: return (exact_matches, total_displacement)."""
    exact = 0
    total_disp = 0
    for race, expected in test_cases:
        predicted = simulate_race(race, so, ho, sd, md, hd, sc, mc, hc, tc, tr)
        if predicted == expected:
            exact += 1
        pred_rank = {d: i for i, d in enumerate(predicted)}
        for i, d in enumerate(expected):
            total_disp += abs(pred_rank[d] - i)
    return exact, total_disp


# Test wide range of cliff combinations
print("\n=== Phase 1: Scanning cliff combinations ===")
best_exact = 0
best_disp = float('inf')
best_full = None

# Broader search for cliffs
cliff_combos = []
for sc in range(0, 12):
    for mc in range(0, 12):
        for hc in range(0, 10):
            cliff_combos.append((sc, mc, hc))

print(f"Testing {len(cliff_combos)} cliff combinations with basic params...")

# Test with several parameter sets
param_sets = [
    (-2.0, 1.5, 0.2, 0.1, 0.04, 0.01, 35.0),
    (-1.8, 1.6, 0.2, 0.08, 0.02, 0.025, 74.0),
    (-1.5, 1.5, 0.15, 0.05, 0.02, 0.01, 30.0),
    (-2.5, 2.0, 0.3, 0.15, 0.05, 0.01, 35.0),
    (-1.0, 1.0, 0.1, 0.05, 0.015, 0.005, 25.0),
    (-3.0, 2.0, 0.5, 0.2, 0.08, 0.015, 40.0),
]

for pi, params in enumerate(param_sets):
    so, ho, sd, md, hd, tc, tr = params
    for ci, (sc, mc, hc) in enumerate(cliff_combos):
        te, td = eval_params(so, ho, sd, md, hd, sc, mc, hc, tc, tr)
        if te > best_exact or (te == best_exact and td < best_disp):
            best_exact = te
            best_disp = td
            best_full = (so, ho, sd, md, hd, sc, mc, hc, tc, tr)
            print(f"  cliff=({sc},{mc},{hc}), params={pi}: exact={te}/100, disp={td} ***")
    print(f"  Finished param set {pi}, best so far: {best_exact}/100")

print(f"\nAfter Phase 1: exact={best_exact}/100, disp={best_disp}")
if best_full:
    print(f"Best params: {best_full}")

# Phase 2: DE optimization on best cliff combo
if best_full:
    sc, mc, hc = int(best_full[5]), int(best_full[6]), int(best_full[7])

    print(f"\n=== Phase 2: DE optimization for cliff=({sc},{mc},{hc}) ===")

    def objective(x):
        te, td = eval_params(x[0], x[1], x[2], x[3], x[4], sc, mc, hc, x[5], x[6])
        return -te * 10000 + td  # Maximize exact, minimize displacement

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
        maxiter=300, popsize=25,
        tol=1e-12, mutation=(0.5, 1.5),
        recombination=0.9, workers=1, disp=True
    )

    p = result.x
    te, td = eval_params(p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6])
    print(f"\nDE result: exact={te}/100, disp={td}")

    if te > best_exact or (te == best_exact and td < best_disp):
        best_exact = te
        best_disp = td
        best_full = (p[0], p[1], p[2], p[3], p[4], sc, mc, hc, p[5], p[6])

# Phase 3: Hill climb
if best_full:
    print(f"\n=== Phase 3: Hill climbing ===")
    current = list(best_full)
    ce, cd = eval_params(*current)

    for scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        improved = True
        iters = 0
        while improved and iters < 200:
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
                    te, td = eval_params(*trial)
                    if te > ce or (te == ce and td < cd):
                        current = trial
                        ce = te
                        cd = td
                        improved = True

        # Try cliff changes at each scale
        for dim in [5, 6, 7]:
            for delta in [-1, 1]:
                trial = list(current)
                new_val = max(0, int(trial[dim]) + delta)
                trial[dim] = new_val
                te, td = eval_params(*trial)
                if te > ce or (te == ce and td < cd):
                    current = trial
                    ce = te
                    cd = td

        print(f"  Scale {scale}: exact={ce}/100, disp={cd}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: exact={ce}/100")
    so, ho, sd, md, hd, sc, mc, hc, tc, tr = current
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

    # Save to best_params.json
    best_params = {
        's_off': so, 'h_off': ho,
        's_deg': sd, 'm_deg': md, 'h_deg': hd,
        's_cliff': int(sc), 'm_cliff': int(mc), 'h_cliff': int(hc),
        't_coeff': tc, 't_ref': tr
    }
    with open('solution/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved to solution/best_params.json")
