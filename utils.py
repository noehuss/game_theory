import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Colormap
colors = ["#384E77","#8BBEB2","#E6F9AF","pink", "limegreen","black","orange","grey","maroon"]  # R -> G -> B


def plot_merit_order(df:pd.DataFrame, demand:int, strategic_producer=None, sp_profit=None, ax_fig=None):
    """
    Use a dataframe with:
    production: quantities produced by each producer
    bids: Bids of each producer
    producer: Producer name
    capacities: Capacities of each generator
    """
    colors = ["#384E77","#8BBEB2","#E6F9AF","pink","limegreen","black","orange","grey","maroon"]
    df["color"] = pd.Series(colors[:len(df)])
    df["xpos"] = ""
    df.sort_values(by="bids", inplace=True)
    df["cumulative_capa"] = df["capacities"].cumsum()
    
    for index in df.index:
        print(df.index)
        i = df.index.get_loc(index)
        print(i)
        if i == 0:
            df.loc[index, "xpos"] = df.at[index, 'capacities']/2
        else:
            print(index)
            df.loc[index, "xpos"] = df.at[index, 'capacities']/2 + df.iloc[i-1].at["cumulative_capa"]

    def cut_off(demand):
        #To get the cutoff power plant 
        for index in df.index:

            if df.loc[index, "cumulative_capa"] < demand:
                pass

            else:
                cut_off_power_plant = index
                print ("Power plant that sets the electricity price is: ", cut_off_power_plant)
                return cut_off_power_plant

    print(df)               
    plt.rcParams["font.size"] = 16

    xpos = df['xpos'].values.tolist()
    y = df['bids'].values.tolist()

    #width
    w = df['capacities'].values.tolist()
    cut_off_power_plant = cut_off(demand)

    if ax_fig is None:
        fig, ax = plt.subplots(figsize=(16,10))
    else:
        ax = ax_fig

    ax.bar(xpos, height=y, width=w, fill=True, color=df["color"].tolist())

    ax.set_xlim(0, df["capacities"].sum()+10)
    ax.set_ylim(0, df['bids'].max()+20)

    ax.hlines(y=df.at[cut_off_power_plant, 'bids'],
               xmin=0,
               xmax=demand,
               color='gray',
               linestyle='dashed')
    
    ax.axvline(x=demand,
               color='gray',
               linestyle='dashed')
    
    text = ax.text(x = demand - df.at[cut_off_power_plant, "production"]/2,
            y = df.at[cut_off_power_plant, "bids"] + 10,
            s = f"Electricity price:\n {df.at[cut_off_power_plant, 'bids']} $/MWh",
            ha = 'center',
            color='gray')
    text.set_bbox(dict(facecolor='white', alpha=1))

    if sp_profit is not None and strategic_producer is not None:
        text_sp = plt.text(x=0.15, y=0.90, s=f"Strategic producer: {strategic_producer}\nProfit: {sp_profit} $", ha='center',  transform=ax.transAxes, color='gray')
        text_sp.set_bbox(dict(facecolor='white', alpha=1))

    ax.set_xlabel("Power plant production (MW)")
    ax.set_ylabel("Bids ($/MWh)")
    ax.legend(ax.patches, df['producer'].to_list(),
              loc = "best")
    
    if ax_fig is None:
        plt.show()
    else:
        return ax

def plot_bids_evolution(df:pd.DataFrame, dict_alphas: dict, single_graph=True, market_prices=None):
    colors = ["#384E77","#8BBEB2","#E6F9AF","pink","limegreen","black","orange","grey","maroon"]
    df["color"] = pd.Series(colors[:len(df)])
    df_alphas =pd.DataFrame(dict_alphas)
    if single_graph:
        fig, ax = plt.subplots(figsize=(16,10))
        for i,producers in enumerate(df['producers']):
            ax.plot(df_alphas.columns, df_alphas.iloc[i], label =f'Bids of {producers}', marker ='o', color=df.loc[i, 'color'])
            ax.axhline(df.at[i,'minFuelCosts'], color=df.loc[i, 'color'], linestyle='--', label=f'Truthful cost \nof producer {producers}')
            ax.set_xlabel('Number of iteration')
            ax.set_ylabel('Prices €/MWh')
            ax.grid(True, alpha=0.6, linestyle='--')
            ax.legend(loc='best')
        if market_prices is not None:
            ax.plot(df_alphas.columns.to_list()[1:], market_prices[1:], marker='x', linestyle='None', color='black', label='Market Price', markersize=12)
        plt.show()
    else:
        fig, axes = plt.subplots(1, len(df['producers']), figsize=(15, 5))
        for i,producers in enumerate(df['producers']):
            axes[i].plot(df_alphas.columns, df_alphas.iloc[i], label =f'Bids of {producers}', marker ='o', color=df.loc[i, 'color'])
            axes[i].axhline(df.loc[i,'minFuelCosts'], color=df.loc[i, 'color'], linestyle='--', label=f'Truthful cost \nof producer {producers}')
            axes[i].set_xlabel('Number of iteration')
            axes[i].set_ylabel('Prices €/MWh')
            axes[i].grid(True, alpha=0.6, linestyle='--')
            axes[i].legend(loc='best')
        plt.tight_layout()
        plt.show()
    return df_alphas


def plot_dispatch_br_mc(results_mc:pd.DataFrame, results_br:pd.DataFrame, demand):
    """Plot the dispatch of each generator at (i) the converged equilibrium and
    (ii) the centralized optimum obtained by clearing with true costs (“central
    dispatch”). Use bar charts."""
    df = [results_mc, results_br]
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    for i, ax in enumerate(axes):
        ax = plot_merit_order(df=df[i], demand=demand, ax_fig=ax)
    plt.show()
    