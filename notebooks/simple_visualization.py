import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def shot_type_analysis(df: pd.DataFrame, years: list = [], save_plot: bool = True):
    '''
    Returns a bar plot of the number of shots and goals with the ratio of the two grouped by shot type.
    You can specify the year(s) of the season you want
    '''
    # removing "fake" shots
    types_to_exclude = ["Deflected", "Tip-In", "Wrap-around"]

    # filter df given the specified years
    if years:
        df = df[df["year"].isin(years)]

    # we exclude those
    filtered_data = df[~df["typeDeTir"].isin(types_to_exclude)]

    # grouping by shot type
    grouped_data = filtered_data.groupby("typeDeTir")

    # nb of shots and goals for each shot type
    total_shots = grouped_data.size()
    total_goals = grouped_data["eventTypeId"].apply(lambda x: (x == "GOAL").sum())

    # success rate
    success_rate = (total_goals / total_shots).round(2)

    # creating the bar plot
    bar_width = 0.8
    index = range(len(total_shots))
    plt.figure(figsize=(12, 6))
    plt.bar(index, total_shots, bar_width, label="Total Shots", color="teal")
    plt.bar(index, total_goals, bar_width, label="Total Goals", bottom=total_shots, color="deeppink")  # "bottom" is used to stack the bars

    # we add here success rates above each bar
    for i, rate in enumerate(success_rate):
        plt.text(index[i], total_shots[i] + total_goals[i], f"{rate}", ha="center", va="bottom", fontsize=14)

    # title and labels
    plt.xlabel("Shot Type")
    plt.xticks(index, total_shots.index, rotation=45)
    plt.legend()
    plt.title("Comparison of Shot Types and Goals in a Hockey Season with Success Rate for year(s) : " + str(years) , fontsize=14)
    plt.ylabel("Number of Shots or Goals")

    # display the plot
    plt.tight_layout()
    plt.show()

    if save_plot:
        # save the figure in /figures 
            plt.savefig('shot_goal.png', format='png', dpi=300)

    return success_rate