import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(mp_br, prod_df, marginal_costs):

    producers = prod_df['producers'].tolist()
    T = len(mp_br.price_time)

    # Prepare arrays
    prices = mp_br.price_time
    dispatch = np.array(mp_br.dispatch_time)      # shape (T, G)
    bids = np.array(mp_br.bids_time)             # shape (T, G)

    mc = np.array(marginal_costs)                # shape (G,)

    # -----------------------------
    # 1. Price over time
    # -----------------------------
    plt.figure(figsize=(7,4))
    plt.plot(range(T), prices, marker='o')
    plt.title("Market-Clearing Price Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2. Dispatch by producer
    # -----------------------------
    plt.figure(figsize=(8,5))
    for i, prod in enumerate(producers):
        plt.plot(range(T), dispatch[:,i], marker='o', label=prod)
    plt.title("Dispatch (MW) by Producer Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Dispatch MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3. Bids by producer
    # -----------------------------
    plt.figure(figsize=(8,5))
    for i, prod in enumerate(producers):
        plt.plot(range(T), bids[:,i], marker='o', label=prod)
    plt.title("Strategic Bids by Producer Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Bid (€/MWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Profit by producer per hour
    # -----------------------------
    profits = np.zeros((T, len(producers)))
    for t in range(T):
        for i in range(len(producers)):
            q = dispatch[t,i]
            p = prices[t]
            c = mc[i]
            profits[t,i] = p*q - c*q

    plt.figure(figsize=(8,5))
    for i, prod in enumerate(producers):
        plt.plot(range(T), profits[:,i], marker='o', label=prod)
    plt.title("Profit per Producer per Hour")
    plt.xlabel("Time step")
    plt.ylabel("Profit (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 5. Optional: Ramping-limited Pmax curves
    # -----------------------------
    if hasattr(mp_br, "dispatch_time"):
        # To do this, you must store Pmax_t in mp_br.
        # If you want, I can extend your class to store these.
        pass
