"""
This script allows one to run DVM on the command line.
"""

from datetime import datetime
import json
import os
import pandas as pd
import sys

import dvm
import dvm_plot
import elect
import tools


def is_int(var):
    try:
        int(var)
        return True
    except ValueError:
        return False


def is_float(var):
    try:
        float(var)
        return True
    except ValueError:
        return False


def user_config():
    """
    Gather user input to configure the DVM run.
    """
    print("***** Election Choice *****\n")

    print("The following elections are available for analysis.")
    print("1) Chicago Mayoral 2011")
    print("2) Chicago Mayoral 2015")
    print("3) Chicago Mayoral 2019")
    election_choice = input("Please input the number of the election you would like to analyze: ")
    while election_choice not in ['1', '2', '3']:
        election_choice = input("Please input either 1, 2, or 3: ")
    print()

    print("***** Model Setup *****\n")

    phc_granularity = input("How granular should the PHC be? Please input an integer between 5 and 1000: ")
    while not is_int(phc_granularity) or int(phc_granularity) > 1000 or int(phc_granularity) < 5:
        phc_granularity = input("Please input an integer between 5 and 1000: ")
    print()

    print("***** Chain Logistics *****\n")

    print("Which form of scoring would you like to use?")
    print("1) Probability-based")
    print("2) Expectation-based")
    scoring_choice = input("Please input the number of the scoring type you would like to use: ")
    while scoring_choice not in ['1', '2']:
        scoring_choice = input("Please input either 1 or 2: ")
    print()

    if scoring_choice == '1':
        expec_scoring = False
    else:
        expec_scoring = True

    print("Which kernel would you like to use?")
    print("1) Random Walk Metropolis")
    print("2) Hamiltonian Monte Carlo")
    kernel_choice = input("Please input the number of the kernel you would like to use: ")
    while kernel_choice not in ['1', '2']:
        kernel_choice = input("Please input either 1 or 2: ")
    print()

    if kernel_choice == '1':
        hmc = False
    else:
        hmc = True

    n_steps = input("For how many steps should the chain be run? Please input an integer between 10 and 10000000: ")
    while not is_int(n_steps) or int(n_steps) > 10000000 or int(n_steps) < 10:
        n_steps = input("Please input an integer between 10 and 10000000: ")
    print()

    burn_frac = input("What fraction of the chain should be burned? Please input a real number between 0 and 1: ")
    while not is_float(burn_frac) or float(burn_frac) > 1 or float(burn_frac) < 0:
        burn_frac = input("Please input a real number between 0 and 1: ")
    print()

    return {
        'election_choice': election_choice,
        'phc_granularity': int(phc_granularity),
        'expec_scoring': expec_scoring,
        'hmc': hmc,
        'n_steps': int(n_steps),
        'burn_frac': float(burn_frac)
    }


def main():
    config = user_config()

    # Import data and clip to two candidates and three races
    demo = pd.read_csv('../tests/example_electoral_data/chicago_demo.csv')
    demo_clip = demo.drop(['asian', 'other'], axis=1)

    if config['election_choice'] == '1':
        elec_data = pd.read_csv('../tests/example_electoral_data/chi_mayor_2011.csv')
        elec_data_clip = elec_data[['prec_id', 'Emanuel', 'DelValle']]
    elif config['election_choice'] == '2':
        elec_data = pd.read_csv('../tests/example_electoral_data/chi_mayor_2015.csv')
        elec_data_clip = elec_data[['prec_id', 'Emanuel', 'Garcia']]
    else:
        elec_data = pd.read_csv('../tests/example_electoral_data/chi_mayor_2019.csv')
        elec_data_clip = elec_data[['prec_id', 'Lightfoot', 'Preckwinkle']]

    # Separate by ward, instead of by precinct
    # Demographic data
    demo_clip['ward_id'] = demo_clip['prec_id'].str[:3].str.replace('C', 'W')
    demo_ward = demo_clip.groupby('ward_id').agg('sum')

    # Election data
    elec_data_clip['ward_id'] = elec_data_clip['prec_id'].str[:3].str.replace('C', 'W')
    elec_data_ward = elec_data_clip.groupby('ward_id').agg('sum')

    # Create election
    election = elect.create_elections(elec_data_ward, demo_ward, "election_" + config['election_choice'], id='ward_id')

    # Run election
    dvm_results = dvm.dvm_elections(election, phc_granularity=config['phc_granularity'], hmc=config['hmc'], expec_scoring=config['expec_scoring'], burn_frac=config['burn_frac'], n_steps=config['n_steps'], verbose=True)

    # Get summary plots
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directory = "experiments/" + current_time + '/'

    try:
        os.mkdir(directory)
    except OSError:
        print(f"Failed to create a directory at {directory}.")
        print("Exiting.")
        sys.exit()

    # Trace plot
    dvm_plot.trace_plot(dvm_results['chain_results'], save=True, filename=directory + 'trace_plot')

    # Mean PHC
    mean_phc = dvm.mean_phc(dvm_results['chain_results'])
    dvm_plot.phc_plot_3d(mean_phc, election.dpp['W01'], save=True, filename=directory + 'mean_phc')

    # MLE PHC
    mle_phc = dvm.chain_mle(dvm_results['chain_results'])[0]
    dvm_plot.phc_plot_3d(mle_phc, election.dpp['W01'], save=True, filename=directory + 'mle_phc')

    # Calculate vote percentages
    mle_vote_pcts = elect.get_vote_pcts(tools.get_most_probable_cell(mle_phc), config['phc_granularity'], election.dpp)

    numpy_mle_vote_pcts = {}
    for ward in mle_vote_pcts:
        numpy_mle_vote_pcts[ward] = {}
        for group, tensor in mle_vote_pcts[ward].items():
            numpy_mle_vote_pcts[ward][group] = str(tensor.numpy())

    try:
        with open(directory + "mle_vote_pcts.json", "w") as file:
            json.dump(numpy_mle_vote_pcts, file)
    except:
        print(f"Unable to write {directory}mle_vote_pcts.json")

    mean_vote_pcts = elect.get_vote_pcts(tools.get_most_probable_cell(mean_phc), config['phc_granularity'], election.dpp)

    numpy_mean_vote_pcts = {}
    for ward in mean_vote_pcts:
        numpy_mean_vote_pcts[ward] = {}
        for group, tensor in mean_vote_pcts[ward].items():
            numpy_mean_vote_pcts[ward][group] = str(tensor.numpy())

    try:
        with open(directory + "mean_vote_pcts.json", "w") as file:
            json.dump(numpy_mean_vote_pcts, file)
    except:
        print(f"Unable to write {directory}mean_vote_pcts.json")

    print()
    print("***** DVM analysis complete *****")
    print(f"The results are in {directory}")
    print("Bye.")


if __name__ == "__main__":
    main()
