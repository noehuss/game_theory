import pandas as pd
import matplotlib.pyplot as plt

def plot_merit_order(df:pd.DataFrame, demand:int, strategic_producer:str):
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
    df["cumulative_prod"] = df["production"].cumsum()
    
    for index in df.index:
        print(df.index)
        i = df.index.get_loc(index)
        print(i)
        if i == 0:
            df.loc[index, "xpos"] = df.at[index, 'capacities']/2
        else:
            print(index)
            df.loc[index, "xpos"] = df.at[index, 'capacities']/2 + df.iloc[i-1].at["cumulative_prod"]

    def cut_off(demand):
        #To get the cutoff power plant 
        for index in df.index:

            if df.loc[index, "cumulative_prod"] < demand:
                pass

            else:
                cut_off_power_plant = index
                print ("Power plant that sets the electricity price is: ", cut_off_power_plant)
                return cut_off_power_plant

    print(df)               
    plt.figure(figsize=(20, 12))
    plt.rcParams["font.size"] = 16

    xpos = df['xpos'].values.tolist()
    y = df['bids'].values.tolist()

    #width
    w = df['capacities'].values.tolist()
    cut_off_power_plant = cut_off(demand)

    fig = plt.bar(xpos, height=y, width=w, fill=True, color=df["color"].tolist())

    plt.xlim(0, df["capacities"].sum())
    plt.ylim(0, df['bids'].max()+20)

    plt.hlines(y=df.at[cut_off_power_plant, 'bids'],
               xmin=0,
               xmax=demand,
               color='gray',
               linestyle='dashed')
    
    plt.axvline(x=demand,
               color='gray',
               linestyle='dashed')
    
    plt.text(x = demand - df.at[cut_off_power_plant, "production"]/2,
            y = df.at[cut_off_power_plant, "bids"] + 5,
            s = f"Electricity price:\n {df.at[cut_off_power_plant, 'bids']} $/MWh",
            ha = 'center',
            color='gray')


    plt.xlabel("Power plant production (MW)")
    plt.ylabel("Bids ($/MWh)")
    plt.legend(fig.patches, df['producer'].to_list(),
              loc = "best")
    plt.show()