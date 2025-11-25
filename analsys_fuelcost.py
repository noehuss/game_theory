import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from best_response import BR
from market_clearing import MarketClearing


# ------------------------------------------------------------
# 1. HELPER: BUILD PRODUCER DATAFRAMES
# ------------------------------------------------------------

def build_symmetric_system(num_players: int,
                           capacity_per_unit: float,
                           c_lower: float,
                           c_upper: float) -> pd.DataFrame:
    """
    Build a symmetric system with 'num_players' generators.
    All have the same capacity and the same fuel cost bounds [c_lower, c_upper].
    """
    producers = [f"P{i+1}" for i in range(num_players)]
    capacities = [capacity_per_unit] * num_players
    min_costs = [c_lower] * num_players
    max_costs = [c_upper] * num_players

    return pd.DataFrame({
        "producers": producers,
        "capacities": capacities,
        "minFuelCosts": min_costs,
        "maxFuelCosts": max_costs,
    })


# ------------------------------------------------------------
# 2. CORE: SEARCH WORST INEFFICIENCY FOR A GIVEN SYSTEM
# ------------------------------------------------------------

def search_worst_inefficiency(prod_df: pd.DataFrame,
                              demand: float,
                              nb_segments: int = 3,
                              max_iter: int = 100,
                              tau_alphas: float = 1.0,
                              tolerance: float = 0.02,
                              plot_hist: bool = False):
    """
    For a fixed set of producers (prod_df) and a fixed demand,
    iterate over a bounded discrete grid of marginal costs (c_Gi values)
    and compute the Price of Anarchy (PoA = SC_eq / SC_opt) for each point.

    Returns:
        results: dict with
            - "worst_poa": max PoA over the grid
            - "avg_poa": average PoA over the grid
            - "convergence_rate": share of instances that converged
            - "poa_values": list of all PoA values
            - "converged_flags": list of booleans, per grid point
            - "thetas": list of marginal cost vectors tested
    """
    nb_producers = len(prod_df)

    # Build discrete grid of marginal costs for each producer
    segments_per_producer = []
    for i in range(nb_producers):
        grid_i = np.linspace(
            prod_df["minFuelCosts"].iloc[i],
            prod_df["maxFuelCosts"].iloc[i],
            num=nb_segments
        )
        segments_per_producer.append(list(grid_i))

    thetas = [list(theta) for theta in product(*segments_per_producer)]

    poa_values = []
    converged_flags = []

    for theta in thetas:
        theta = list(theta)  # ensure list type

        # --- Best response algorithm ---
        br = BR(
            bids_init=theta,
            marginal_costs=theta,
            demand=demand,
            prod_df=prod_df,
            tolerance=tolerance,
        )
        br.run_BR(nb_iter=max_iter, tau_alphas=tau_alphas)
        converged = br.convergenceReached()
        converged_flags.append(converged)

        # Equilibrium dispatch (from BR)
        q_eq = np.array(br.get_equilibrium_dispatch(), dtype=float)
        theta_arr = np.array(theta, dtype=float)
        SC_eq = float(np.dot(theta_arr, q_eq))

        # Socially optimal dispatch (truthful bids = costs)
        mc_opt = MarketClearing(
            bids=theta,
            marginal_costs=theta,
            demand=demand,
            prod_df=prod_df
        )
        q_opt = np.array(mc_opt.get_dispatch(), dtype=float)
        SC_opt = float(np.dot(theta_arr, q_opt))

        PoA = SC_eq / SC_opt if SC_opt > 0 else np.nan
        poa_values.append(PoA)

    # --- Aggregate statistics ---
    poa_values_clean = [p for p in poa_values if not np.isnan(p)]
    worst_poa = max(poa_values_clean) if poa_values_clean else np.nan
    avg_poa = float(np.mean(poa_values_clean)) if poa_values_clean else np.nan
    convergence_rate = sum(converged_flags) / len(converged_flags)

    # Optional: histogram of PoA values for this system
    if plot_hist and poa_values_clean:
        plt.figure(figsize=(7, 5))
        plt.hist(poa_values_clean, bins=20, edgecolor="black")
        plt.xlabel("Price of Anarchy (SC_eq / SC_opt)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of PoA over fuel-cost grid "
                  f"(|I| = {len(prod_df)})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    results = {
        "worst_poa": worst_poa,
        "avg_poa": avg_poa,
        "convergence_rate": convergence_rate,
        "poa_values": poa_values,
        "converged_flags": converged_flags,
        "thetas": thetas,
    }
    return results


# ------------------------------------------------------------
# 3. EXERCISE 4 – PART 1:
#    THREE-PLAYER SYSTEM (HETEROGENEOUS SET-UP)
# ------------------------------------------------------------

def run_three_player_analysis(nb_segments: int = 3):
    """
    Exercise 4 – first bullet:
    For a system with exactly three players, compute the worst equilibrium
    inefficiency over a bounded discrete grid of c_Gi values per generator.
    """
    # This is the "convergent set-up" from Exercise 3 (example):
    prod_df_3 = pd.DataFrame({
        "producers": ["P1", "P2", "P3"],
        "capacities": [100, 40, 40],
        "minFuelCosts": [10, 30, 35],
        "maxFuelCosts": [15, 60, 70],
    })

    # Demand chosen as in your earlier script (e.g. 120 MW)
    demand_3 = 120.0

    print("=== Three-player system analysis ===")
    print(f"Fuel cost grid with {nb_segments} segments per player.")
    print(prod_df_3)

    results_3 = search_worst_inefficiency(
        prod_df=prod_df_3,
        demand=demand_3,
        nb_segments=nb_segments,
        max_iter=100,
        tau_alphas=1.0,
        tolerance=0.02,
        plot_hist=True,  # show PoA distribution for the report
    )

    print(f"Worst PoA (3 players): {results_3['worst_poa']:.4f}")
    print(f"Average PoA (3 players): {results_3['avg_poa']:.4f}")
    print(f"Convergence rate over grid (3 players): "
          f"{100*results_3['convergence_rate']:.1f}%")

    return results_3


# ------------------------------------------------------------
# 4. EXERCISE 4 – PART 2:
#    SYSTEMS WITH |I| = 4,...,10
# ------------------------------------------------------------

def run_scaling_analysis(min_players,
                         max_players,
                         capacity_per_unit: float,
                         c_lower: float,
                         c_upper: float,
                         nb_segments: int,
                         load_factor: float):
    """
    Exercise 4 – second bullet:
    Repeat the analysis for systems with |I| in {4, ..., 10} players.

    Assumptions (to state clearly in the report):
      - Each generator has identical capacity 'capacity_per_unit'.
      - Total capacity = |I| * capacity_per_unit.
      - Demand scales linearly with |I| as:
            demand = load_factor * total_capacity
        so that system loading (e.g. 55%) is constant as |I| changes.
      - All players share identical fuel cost bounds [c_lower, c_upper].
    """
    num_players_list = []
    worst_poa_list = []
    convergence_list = []

    for n in range(min_players, max_players + 1):
        prod_df_n = build_symmetric_system(
            num_players=n,
            capacity_per_unit=capacity_per_unit,
            c_lower=c_lower,
            c_upper=c_upper,
        )

        total_capacity = prod_df_n["capacities"].sum()
        demand_n = load_factor * total_capacity

        print(f"\n=== Analysis for |I| = {n} players ===")
        print(f"Total capacity = {total_capacity:.1f}, demand = {demand_n:.1f}")

        results_n = search_worst_inefficiency(
            prod_df=prod_df_n,
            demand=demand_n,
            nb_segments=nb_segments,
            max_iter=100,
            tau_alphas=1.0,
            tolerance=0.02,
            plot_hist=False,  # avoid too many plots
        )

        num_players_list.append(n)
        worst_poa_list.append(results_n["worst_poa"])
        convergence_list.append(results_n["convergence_rate"])

        print(f"Worst PoA (|I|={n}): {results_n['worst_poa']:.4f}")
        print(f"Convergence rate (|I|={n}): "
              f"{100*results_n['convergence_rate']:.1f}%")

    # Convert to DataFrame for easier use
    df_results = pd.DataFrame({
        "num_players": num_players_list,
        "worst_poa": worst_poa_list,
        "convergence_rate": convergence_list,
    })

    return df_results


# ------------------------------------------------------------
# 5. PLOTTING FUNCTIONS FOR THE ASSIGNMENT
# ------------------------------------------------------------

def plot_inefficiency_vs_players(df_results: pd.DataFrame):
    """
    Plot (a) Equilibrium inefficiency as a function of |I|.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(df_results["num_players"],
             df_results["worst_poa"],
             marker="o")
    plt.xlabel("Number of players |I|")
    plt.ylabel("Worst equilibrium inefficiency (PoA)")
    plt.title("Worst Price of Anarchy vs Number of Players")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_convergence_vs_players(df_results: pd.DataFrame):
    """
    Plot (b) Share (%) of instances that converged as a function of |I|.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(df_results["num_players"],
             100 * df_results["convergence_rate"],
             marker="s")
    plt.xlabel("Number of players |I|")
    plt.ylabel("Converged instances [%]")
    plt.title("Share of Converged Instances vs Number of Players")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. MAIN: RUN EVERYTHING
# ------------------------------------------------------------

if __name__ == "__main__":
    # 1) Three-player heterogeneous system
    results_3 = run_three_player_analysis(nb_segments=3)

    # 2) Scaling analysis for |I| = 4,...,10
    df_scaling = run_scaling_analysis(
        min_players=4,
        max_players=6,
        capacity_per_unit=100.0,
        c_lower=10.0,
        c_upper=50.0,
        nb_segments=3,
        load_factor=0.55,
    )

    # 3) Required plots for the report
    plot_inefficiency_vs_players(df_scaling)
    plot_convergence_vs_players(df_scaling)
