
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_transition_matrix(column_vals, nr_antecedents):
    
    """
    Calculates the transition probability matrix from a sequence of states. 

    Args: 
        column_vals: the sequence of data
        nr_antecedents: the number of samples in the sequence that are used to compute the transition probabilities. 1 is a 1st order chain etc..

    Returns: 
        states (strings): the unique states in the system
        unique_antecedents (list of strings): the unique samples or groups of samples (if nr_antecedents >1) that are used to calculate the probabilities.
        encoding_dict: dictionary that encodes the string-states into numbers
        trans_matrix: the transition probability matrix
    """

    states = list(set(column_vals))

    #Encode the different options that appear in the chosen sequence into numbers. 
    encoding_dict = {state: idx for idx, state in enumerate(states)}
    vals_encoded = [encoding_dict[val] for val in column_vals]

    #The window size is one element bigger than the nr_antecedents to include the follow-up element as well.
    window_size = nr_antecedents+1

    #Group all segments of the sequence of the chosen window size into an array (with the number of columns equal to the window-size).
    consecutives = np.vstack([vals_encoded[i: i+window_size] for i in range(len(vals_encoded)-window_size)])

    #Extract the unique antecedent groups - if nr_antecedents is 1, this is equal to the unique states.
    all_antecedents = consecutives[:, :nr_antecedents]
    unique_antecedents = np.unique(all_antecedents, axis = 0)

    #Initialize empty transition probability matrix, with each row belonging to an antecedent group and each column to a unique state. 
    trans_matrix = np.zeros((len(unique_antecedents), len(states)))

    for i, antecedent in enumerate(unique_antecedents): 

        #filter the array for all entries where the antecedent appears first
        antecedent_first = consecutives[np.all(all_antecedents == antecedent, axis = 1)]
        
        #Extract all "follow-up" states that appear after this antecedent, and count the times each one appears.
        follow_up_counts = pd.Series(antecedent_first[:, -1]).value_counts()

        #The values of these "follow-up" states correspond to the column-indeces of the transition matrix, so we can directly fill the antecedent row with their counts.
        trans_matrix[i, follow_up_counts.index] = follow_up_counts.values

        #Turn the counts into probabilities by dividing each count by the total number of cases where this antecedent appears first. Equivalent to dividing each row-item by the total row-sum.
        trans_matrix[i] = trans_matrix[i]/len(antecedent_first)

    #Check that each row adds up to one, or zero within a certain tolerance (in the case where the sample is last in the sequence and only appears then)
    rowsums = trans_matrix.sum(axis = 1).astype(int)
    if not np.all(np.isclose(rowsums, 1) | np.isclose(rowsums, 0)):
        raise ValueError("All rows of probabilities must sum to 1.0 or 0.0")
 

    decoding_dict = {value: key for key, value in encoding_dict.items()}
    
    #decode the numerical unique_antecedents back into the original state-strigns
    unique_antecedents = pd.DataFrame(unique_antecedents).applymap(lambda x: decoding_dict[x])

    #turn each unique antecedent into one string and group into a list
    unique_antecedents = list(unique_antecedents.applymap(lambda x: str(x)).apply(lambda row: " ".join(row), axis = 1).values)

    trans_matrix = pd.DataFrame(trans_matrix, columns = states, index = unique_antecedents)
    
    return  states, unique_antecedents, encoding_dict, trans_matrix

def check_irreducibility(trans_matrix): 
    """
    Checks whether a transition probability matrix is irreducible or not, as this is a condition for it having a stationary distirbution.
    This is done by taking the power of the matrix multiple times (number of states times) - if the final values are all positive, then it is said to be irreducible.

    Args: 
        trans_matrix: the transition probability matrix

    Returns: 
        Boolean: True if it is irreducible, False if not.

    """

    if np.equal(*trans_matrix.shape): 

        P = trans_matrix

        n = P.shape[0] 
        P_powered = P.copy()
        
        for i in range(1, n): 
            P_powered = np.dot(P, P_powered )

        if np.all(P_powered > 0): 
            return True
    else: 
        raise ValueError("The martix needs to be square!")
    
    return False


def get_stationary_distribution(trans_matrix, states):

    """
    Gets stationary distribution by solving (P^T - I)x = 0, where P is the transition probability matrix and x is the stationary distribution.
    Only does so if the matrix is irreducible (and thus has a stationary distribution)

    Args: 
        trans_matrix (array): the probability matrix
        states (strings): the unique states of the system
    Returns: 
        A dictionary containing the stationary distribution, where the keys are the states and the values the percentages.

    """

    irred = check_irreducibility(trans_matrix)
    
    if irred == True: 

        P_T = trans_matrix.T
        LHS = P_T-np.eye(P_T.shape[0])
        LHS = np.vstack([LHS, np.ones(P_T.shape[0])]) #this adds a row of ones to the matrix, to include the normalization constraint (as will then set this equal to 1)

        RHS = np.zeros(P_T.shape[0]+1) #the +1 is to include the normalisation constraint below
        RHS[-1] = 1

        stat_distr = np.linalg.lstsq(LHS, RHS, rcond=None)[0]
        stat_distr_dict = dict(zip(states, stat_distr))

        return stat_distr_dict

    else: 
        raise ValueError("The matrix isn't reducible and so does not have a stationary distribution!")


def generate_chain(trans_matrix, states, unique_antecedents, nr_antecedents, start_sample = ["Clear"], num_samples = 100):
    """
    Generates a Markov Chain from a transition matrix of states and a specified starting sample.
    
    Args: 
        trans_matrix (array): the probability matrix
        states (list of strings): the unique states
        unique_antecedents (list of strings): the unique samples/groups used to calculate the probabilities
        nr_antecedents (int): the number of samples in the sequence that are used to compute the transition probabilities. 1 is a 1st order chain etc..
        start_sample (list of strings): A list of the start(ing) sample(s)
        num_samples = The number of samples wanted in the generated chain

    Returns:
        The Markov chain

    """
    chain = []
    chain.extend(start_sample)

    for i in range(num_samples-nr_antecedents):

        rng = np.random.default_rng()

        #Since the start_sample is either a list containing one sample or various samples, joining the contents together as below will lead to the correct "single" string-state.
        start_sample = " ".join(start_sample)
        
        #get the probability row corresponding to the start_sample
        start_sample_idx = unique_antecedents.index(start_sample)
        state_probs = trans_matrix[start_sample_idx]

        #handle the case where the row of probabilites could sum to 0 due to a sample only appearing last
        while np.sum(state_probs) == 0:
            start_sample = rng.choice(unique_antecedents)
            start_sample_idx = unique_antecedents.index(start_sample)
            state_probs = trans_matrix[start_sample_idx]

        #Generate the new sample, append it to the list and reassign start_sample
        next_sample = rng.choice(states, p=state_probs)
        chain.append(next_sample)
        start_sample = chain[-nr_antecedents:]
    
    return chain

def continous_to_discrete(cont_vals, nr_states): 
    """
    Function that turns continuous variable data into discrete data by splitting a spread out range of values into equally-sized bins.

    Args: 
        cont_vals (array) the values we want to bin
        nr_states (int): how many "bins" or states we want to our data to have.
    
    Returns: 
        discrete_vals: A list of the the new values, which are now string-numbers according to which bin-range they fall into.
        ranges: A DataFrame of the original ranges used to bin the data.
    """

    bin_bounds =np.linspace(cont_vals.min(), cont_vals.max(),nr_states + 1)

    discrete_vals = np.digitize(cont_vals, bin_bounds, right = True)
    discrete_vals = [str(val) for val in discrete_vals]

    ranges = np.vstack([bin_bounds[:-1], bin_bounds[1:]]).T
    ranges = pd.DataFrame(ranges, columns=["Start", "End"])
    ranges.index.name = "Range"
    return discrete_vals, ranges

def mcmc_analysis(loc_dfs, column, nr_antecedents, num_repeats, num_samples):

    """
    Main MCMC function that returns results for a specified data column of the dataframes of all locations (if more than one is given).

    Args: 
        loc_dfs: The dictionary of location-data DataFrames
        column (string): The column of the DataFrame that want to be analysed.
        nr_antecedents: the number of samples in the sequence that are used to compute the transition probabilities. 1 is a 1st order chain etc..
        num_repeats: How many chains want to be generated
        num_samples: How many samples we want the generated chain to have


    Returns: 
        loc_trans_matrices: Dictionary that holds a transition matrix (array) for every location
        stat_distributions: Dictionary that holds the stationary distributinos (dicts) for every location
        loc_generated_freqs: Dictionary that holds the generated state frequencies (array) for every location
        loc_generated_chains: Dictionary that holds an array of chains for every location
        loc_states: Dictionary that holds the unique states for every location
    """
    
    loc_trans_matrices = {}
    stat_distributions = {}
    loc_generated_chains = {}
    loc_states = {}
    loc_generated_freqs = {}

    #Iterate through each location: 
    for loc, df in loc_dfs.items(): 
        
        #Retrieve the sequence for the specified column
        col_vals = df[column]
        cont = len(set(col_vals)) > 100 # check whether the data is continuous (set to mean that it has over 100 unique datapoints) 
        if cont == True: #if it is continuous, then make discrete by grouping it into bins
            col_vals, ranges = continous_to_discrete(col_vals, nr_states=20)

        #Perform transition matrix analysis
        states, unique_antecedents, encoding_dict, trans_matrix_df = get_transition_matrix(col_vals, nr_antecedents=nr_antecedents)
        
        #If the matrix is square (i.e., a 1st order chain was used), get the stationary distribution of the original sequence.
        if nr_antecedents == 1: 
            stat_distr_dict = get_stationary_distribution(trans_matrix_df.values, states)

        #Determine the starting sample or group of samples that will produce the first generated sample.
        if column == "summary": 
            start_sample = ["Clear"]*nr_antecedents
        else: 
            start_sample = [np.random.choice(unique_antecedents)]

        #Initialise arrays to hold the Markov chains, as well as the state frequenciesof the chains.
        chains = np.zeros((num_repeats, num_samples)).astype(int)
        freqs = np.zeros((num_repeats, len(states)))

        #Produce chains for num_repeats amount of times.
        for r in range(num_repeats): 
            
            chain = generate_chain(trans_matrix_df.values, states, unique_antecedents, nr_antecedents=nr_antecedents, start_sample=start_sample, num_samples = num_samples)
            encoded_chain = [encoding_dict[val] for val in chain] #turn the chain into numerical vals to handle better later
            chains[r] = encoded_chain

            #Get the state-frequencies of the generated chain
            u_vals, counts = np.unique(encoded_chain, return_counts = True) #
            freqs[r, u_vals] = counts/num_samples

        #Store the results for each location in a dictionary
        loc_generated_chains[loc] = chains
        loc_generated_freqs[loc] = freqs
        loc_trans_matrices[loc] = trans_matrix_df
        loc_states[loc] = states
        if nr_antecedents == 1: 
            stat_distributions[loc] = stat_distr_dict


    return loc_trans_matrices, stat_distributions, loc_generated_freqs,  loc_generated_chains, loc_states

def analyse_chains(chains, states):

    """
    Performs some analysis on the generated chains.

    Args: 
        chains: an array that hold the chain(s) array(s)
        states: the unique states
    
    Returns:
        means (array): The mean values of every chain - useful for data that was originally continuous.
        std_devs (array): The standard deviation of every chain. 
        state_consecs (array): For every chain, holds the maximum consecutive appearance of each state.
    
    """

    means = chains.mean(axis=-1)
    std_devs = chains.std(axis = -1)


    state_consecs = np.zeros((chains.shape[0], len(states)))
    for c, chain in enumerate(chains): 
        consec_lengths = {s: [] for s in range(len(states))}

        current_length = 1
        for s in range(1, len(chain)): 
            if chain[s] == chain[s-1]: 
                current_length += 1
            else: 
                consec_lengths[chain[s-1]].append(current_length)
                current_length = 1

        consec_lengths[chain[s-1]].append(current_length)

        consec_lengths_max = [np.max(l) if len(l) > 0 else 0 for l in consec_lengths.values()]
        state_consecs[c] = consec_lengths_max
        

    state_consecs = np.nan_to_num(state_consecs)

    return means, std_devs, state_consecs

def plot_transition_matrices(loc_trans_matrices, column):

    """
    Plots the transition matrices for each location in one figure. 
    
    """
    
    fig, axes = plt.subplots(1, len(loc_trans_matrices), figsize= (20, 5))
    fig.suptitle(f"Transition probabilities for {column} states", fontsize = 15)

    for i, (loc, matrix) in enumerate(loc_trans_matrices.items()):
        axes[i].set_title(loc)
        sns.heatmap(matrix, ax = axes[i], annot = True)

def plot_loc_stationary_distributions(stat_distributions, column):

    """
    Plots the stationary distributions as a barplot, for the data of each location. 
    
    """
    fig, ax = plt.subplots(1, figsize= (10, 4))

    width = 0.25
    m = 0
    for loc, distr in stat_distributions.items(): 

        states = distr.keys()
        x = np.arange(len(states))
        offset = width*m
        bars = ax.bar(x+offset, distr.values(), width, label = loc)
        ax.bar_label(bars, labels=[f"{h:.2f}" for h in distr.values()])
        m += 1

    m = 0
    ax.set_xticks(x+width, states)
    ax.set_title(f"Long term probabilities of being in each {column} state (stationary distributions)")
    ax.legend()
    plt.show()

def plot_generated_chains(locs, results, column, repeat_loc = False):

    """
    Plots the generated Markov chains for the data of each location. 
    Or, if repeat_loc is set to True, chains of various runs are plotted for a given location.

    Args: 
        locs: A list of location strings. If repeat_loc = True, then input a list of repeat strings (eg)
        results: The dictionary of MCMC results
        column (string): the data variable
    
    """

    fig, ax = plt.subplots(figsize = (13, len(locs)*2))
    ytick_pos = []
    ytick_labels = []
    height = 0
    chain_nr = 0

    for i, loc in enumerate(locs): 

        states = results[column][-1][loc]
        chain = results[column][-2][loc][chain_nr]

        decoding_dict = {i: s for i, s in enumerate(states)}
        colors =  [f"C{int(s)}" for s in range(len(states))]
        cmap = [colors[state] for state in chain]

        ax.step(range(len(chain)), chain + height, where='mid', color='black', linewidth=1)
        ax.scatter(range(len(chain)), chain + height, s = 50, c= cmap, zorder=3)

        if i == 0:
            for s in np.arange(len(states)): 
                ax.plot([], "o", c = f"C{int(s)}", label = f"{decoding_dict[s]}")
                ax.legend()

        ytick_pos.append(height + (len(states)-1)/ 2)
        
        if repeat_loc == True: 
            ytick_labels.append(f"{locs[0]} {chain_nr}")
            chain_nr += 1
        else: 
            ytick_labels.append(loc)

        height += len(states)


    ax.set_ylim(0, len(states)*len(locs))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Time (hrs)")
    ax.set_title(f"Generated chains of {column} states")
    plt.show()