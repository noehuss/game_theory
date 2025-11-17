import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from best_response import BR
from market_clearing import MarketClearing


# -------------------------------------------------
# 1) Utility: build the θ grid given prod_df ranges
# -------------------------------------------------

def build_theta_grid(prod_df: pd.DataFrame, nb_segments: int):
    """
    Build a list of all theta (marginal cost) combinations,
    using minFuelCosts / maxFuelCosts in prod_df and nb_segments points.
    """
    nb_producers = len(prod_df['producers'])
    segments = []

    for i in range(nb_producers):
        rangeFuel = np.linspace(
            prod_df['minFuelCosts'][i],
            prod_df['maxFuelCosts'][i],
            num=nb_segments,
            dtype=int
        )
        segments.append(list(rangeFuel))

    thetas = [list(theta) for theta in product(*segments)]
    return thetas


# -------------------------------------------------------------
# 2) Core experiment for a fixed prod_df: PoA + convergence
# -------------------------------------------------------------

def evaluate_prod_df(prod_df: pd.DataFrame, demand: float, nb_segments: int = 3,
                     br_max_iter: int = 50, tol: float = 0.02):
    """
    For a fixed prod_df (capacities + cost ranges) and demand:
    - build θ grid
    - run BR for each θ
    - compute PoA = SC_eq / SC_opt
    - collect convergence and iterations

    Returns: dict with detailed lists + summary.
    """
    thetas = build_theta_grid(prod_df, nb_segments)

    list_PoA = []
    list_converged = []
    list_iters = []

    for theta in thetas:
        # run BR
        br = BR(
            bids_init=theta,
            marginal_costs=theta,
            demand=demand,
            prod_df=prod_df,
            tolerance=tol
        )
        br.run_BR(nb_iter=br_max_iter)

        converged = br.convergenceReached()
        list_converged.append(converged)
        list_iters.append(br.iteration)

        # equilibrium dispatch
        q_eq = np.array(br.get_equilibrium_dispatch(), dtype=float)
        theta_arr = np.array(theta, dtype=float)
        SC_eq = float(np.dot(theta_arr, q_eq))

        # social optimum (true-cost dispatch)
        mc_opt = MarketClearing(
            bids=theta,
            marginal_costs=theta,
            demand=demand,
            prod_df=prod_df
        )
        q_opt = np.array(mc_opt.get_dispatch(), dtype=float)
        SC_opt = float(np.dot(theta_arr, q_opt))

        PoA = SC_eq / SC_opt
        list_PoA.append(PoA)

    # summary statistics
    list_PoA_arr = np.array(list_PoA, float)
    list_iters_arr = np.array(list_iters, float)
    share_converged = sum(list_converged) / len(list_converged)

    summary = dict(
        avg_PoA=float(np.mean(list_PoA_arr)),
        max_PoA=float(np.max(list_PoA_arr)),
        min_PoA=float(np.min(list_PoA_arr)),
        share_converged=float(share_converged),
        avg_iters=float(np.mean(list_iters_arr)),
        max_iters=int(np.max(list_iters_arr)),
        nb_cases=len(thetas),
    )

    return dict(
        thetas=thetas,
        PoA=list_PoA,
        converged=list_converged,
        iters=list_iters,
        summary=summary,
    )


# ------------------------------------------------------
# 3) Capacity asymmetry experiments
# ------------------------------------------------------

def make_prod_df_cap_asym(base_capacities, base_min_costs, base_max_costs,
                          asym_player_index: int, gamma: float):
    """
    Build a prod_df where ONE player's capacity is scaled by gamma.
    Others stay the same.
    """
    capacities = base_capacities.copy()
    capacities[asym_player_index] *= gamma

    prod_df = pd.DataFrame({
        'producers': [f"P{i+1}" for i in range(len(capacities))],
        'capacities': capacities,
        'minFuelCosts': base_min_costs,
        'maxFuelCosts': base_max_costs,
    })
    return prod_df


def run_capacity_asymmetry_sweep(
    base_capacities, base_min_costs, base_max_costs,
    demand_fraction: float = 0.6,
    nb_segments: int = 3,
    asym_player_index: int = 0,
    gammas=(0.5, 0.8, 1.0, 1.2, 1.5),
    br_max_iter: int = 50,
):
    """
    For each gamma in 'gammas', scale ONE player's capacity and evaluate:
    - average PoA
    - worst-case PoA
    - convergence behaviour
    """
    results = []
    for gamma in gammas:
        prod_df = make_prod_df_cap_asym(
            base_capacities=base_capacities.copy(),
            base_min_costs=base_min_costs.copy(),
            base_max_costs=base_max_costs.copy(),
            asym_player_index=asym_player_index,
            gamma=gamma,
        )

        # Demand proportional to total capacity (like in inefficiencies())
        demand = prod_df['capacities'].sum() * demand_fraction

        out = evaluate_prod_df(
            prod_df=prod_df,
            demand=demand,
            nb_segments=nb_segments,
            br_max_iter=br_max_iter,
        )

        summary = out["summary"]
        summary["gamma"] = gamma
        results.append(summary)

        print(f"gamma={gamma:.2f} -> {summary}")

    # simple visualization: PoA vs gamma, share_converged vs gamma
    gammas = [r["gamma"] for r in results]
    avg_PoA = [r["avg_PoA"] for r in results]
    max_PoA = [r["max_PoA"] for r in results]
    share_conv = [r["share_converged"] for r in results]

    plt.figure()
    plt.plot(gammas, avg_PoA, marker="o", label="avg PoA")
    plt.plot(gammas, max_PoA, marker="s", label="worst PoA")
    plt.xlabel("Capacity scale γ for asymmetric player")
    plt.ylabel("PoA (SC_eq / SC_opt)")
    plt.title("PoA vs capacity asymmetry")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(gammas, share_conv, marker="o")
    plt.xlabel("Capacity scale γ for asymmetric player")
    plt.ylabel("Share of θ-cases that converged")
    plt.title("Convergence vs capacity asymmetry")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

    return results


# ------------------------------------------------------
# 4) Cost asymmetry experiments
# ------------------------------------------------------

def make_prod_df_cost_asym(base_capacities, base_min_costs, base_max_costs,
                           asym_player_index: int, shift: float):
    """
    Build prod_df where ONE player's cost range is shifted by 'shift' (€/MWh).
    Positive shift makes them more expensive, negative shift cheaper.
    """
    min_costs = base_min_costs.copy()
    max_costs = base_max_costs.copy()

    min_costs[asym_player_index] += shift
    max_costs[asym_player_index] += shift

    prod_df = pd.DataFrame({
        'producers': [f"P{i+1}" for i in range(len(base_capacities))],
        'capacities': base_capacities,
        'minFuelCosts': min_costs,
        'maxFuelCosts': max_costs,
    })
    return prod_df


def run_cost_asymmetry_sweep(
    base_capacities, base_min_costs, base_max_costs,
    demand_fraction: float = 0.6,
    nb_segments: int = 3,
    asym_player_index: int = 0,
    shifts=(-10, -5, 0, 5, 10),
    br_max_iter: int = 50,
):
    """
    For each 'shift', change one player's cost range and evaluate
    PoA and convergence.
    """
    results = []
    for shift in shifts:
        prod_df = make_prod_df_cost_asym(
            base_capacities=base_capacities.copy(),
            base_min_costs=base_min_costs.copy(),
            base_max_costs=base_max_costs.copy(),
            asym_player_index=asym_player_index,
            shift=shift,
        )

        demand = prod_df['capacities'].sum() * demand_fraction

        out = evaluate_prod_df(
            prod_df=prod_df,
            demand=demand,
            nb_segments=nb_segments,
            br_max_iter=br_max_iter,
        )

        summary = out["summary"]
        summary["shift"] = shift
        results.append(summary)

        print(f"shift={shift:+.1f} -> {summary}")

    shifts_list = [r["shift"] for r in results]
    avg_PoA = [r["avg_PoA"] for r in results]
    max_PoA = [r["max_PoA"] for r in results]
    share_conv = [r["share_converged"] for r in results]

    plt.figure()
    plt.plot(shifts_list, avg_PoA, marker="o", label="avg PoA")
    plt.plot(shifts_list, max_PoA, marker="s", label="worst PoA")
    plt.xlabel("Cost shift for asymmetric player (€/MWh)")
    plt.ylabel("PoA (SC_eq / SC_opt)")
    plt.title("PoA vs cost asymmetry")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(shifts_list, share_conv, marker="o")
    plt.xlabel("Cost shift for asymmetric player (€/MWh)")
    plt.ylabel("Share of θ-cases that converged")
    plt.title("Convergence vs cost asymmetry")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

    return results


# ------------------------------------------------------
# 5) Example: run from this file
# ------------------------------------------------------

if __name__ == "__main__":
    # Base case similar to your original varying_fuel_costs.py
    base_capacities = [100, 40, 40]
    base_min_costs = [10, 30, 35]
    base_max_costs = [15, 60, 70]

    # ---- capacity asymmetry: change P1's capacity ----
    cap_results = run_capacity_asymmetry_sweep(
        base_capacities=base_capacities,
        base_min_costs=base_min_costs,
        base_max_costs=base_max_costs,
        demand_fraction=0.55,      # like your inefficiencies() function
        nb_segments=3,
        asym_player_index=0,       # make P1 larger/smaller
        gammas=(0.5, 0.8, 1.0, 1.2, 1.5),
        br_max_iter=50,
    )

    # ---- cost asymmetry: make P2 cheaper or more expensive ----
    cost_results = run_cost_asymmetry_sweep(
        base_capacities=base_capacities,
        base_min_costs=base_min_costs,
        base_max_costs=base_max_costs,
        demand_fraction=0.55,
        nb_segments=3,
        asym_player_index=1,       # P2's costs move
        shifts=(-10, -5, 0, 5, 10),
        br_max_iter=50,
    )
