# -*- coding: utf-8 -*-
""" 
We will use this script to process the played games and generate a model to 
predict likelynesss to win
"""
import os,sys,time,random,math,time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.externals import joblib

datadir="./game_state_data/"  #where the game data was saved

def to_string(state, symbols=['1', '2']): #stolen from isolation.py
    """Generate a string representation of the current game state, marking
    the location of each player and indicating which cells have been
    blocked, and which remain open.
    """
    p1_loc = state[-1]
    p2_loc = state[-2]
    
    col_margin = len(str(7 - 1)) + 1
    prefix = "{:<" + "{}".format(col_margin) + "}"
    offset = " " * (col_margin + 3)
    out = offset + '   '.join(map(str, range(7))) + '\n\r'
    for i in range(7):
        out += prefix.format(i) + ' | '
        for j in range(7):
            idx = i + j * 7
            if not state[idx]:
                out += ' '
            elif p1_loc == idx:
                out += symbols[0]
            elif p2_loc == idx:
                out += symbols[1]
            else:
                out += '-'
            out += ' | '
        out += '\n\r'
    
    return out

def load_game(game_file):
    """ 
    Load the state of the game and the winner 
    load the game from a file for later analysis
    each row in the file is the game state.
    row has format [gamestate],player_num_of_winner
    """
    import pickle
    try:
        with open(game_file, 'rb') as f:
            state=pickle.load(f)
    except:
        return False,([0],0)
    return True,state

def load_games_from_dir(datadir):
    """ 
    Attempt to load games from the given datadir, and return them as a 
    single data struct
    """
    #create a list of files 
    try:  
        files =  [x for x in os.listdir(datadir) if x.endswith('.pckl')] #create a list of subdirectories in the current dir
    except:
        sys.stderr.write("expected error->problem removing directories that aren't in list: ignore")
    
    #loading game data from files, and create single list
    games_data=[]
    print("Sample files seen",files[:4])
    for game_file in files[:]:
        load_result,game_data=load_game(datadir+game_file)
        if load_result:
            for turn in game_data[0]:
                turn.append(game_data[1])
                games_data.append(turn)
        else:
            print("failed to load:",game_file)
    # return a dataframe of all the games states and who won
    labels=["cell_"+str(c) for c in range(50)]
    labels.append('player2')
    labels.append('player1')
    labels.append('winner')
    return pd.DataFrame.from_records(games_data,columns=labels)

                
        
        
if __name__ == "__main__":
    game_states_labeled=load_games_from_dir(datadir)
    print(game_states_labeled.info())
    