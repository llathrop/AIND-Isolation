# -*- coding: utf-8 -*-
""" 
We will use this script to process the played games and generate a model to 
predict likelynesss to win
"""
import os,sys,time,random,math

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import train_test_split
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
    for game_file in files[:1000]:
        load_result,game_data=load_game(datadir+game_file)
        if load_result:
            for turn in game_data[0]:
                #(turn[:-1]+1)*-1
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


def grid_search_wrapper(x,y,regr,param,regr_name='BLANK',cachedir="./"):
    start_time = time.time()
    print("In:{}".format(regr))
    filename= 'grid_{}.joblib'.format(regr_name)
    if os.path.isfile(cachedir+filename):
        print(filename," exists, importing ")
        return joblib.load(cachedir+filename) 
    else:
        print("{} not present, running a gridsearch".format(filename))
        #search the param_grid for best params based on the f1 score
        grid_search = GridSearchCV(regr,
                                   param_grid= param,
                                   n_jobs=-1,
                                   scoring=make_scorer(mean_absolute_error,greater_is_better=False)) 
        print("begin gridsearch training")
        grid_search.fit(x,y)
        print("end gridsearch training")
        #reach into the grid search and pull out the best parameters, and set those on the clf
        params={}
        for p in grid_search.best_params_:
            params[p]=grid_search.best_params_[p]
        regr.set_params(**params)
        print("run time:{}s".format(round((time.time()-start_time), 3) ))   
        joblib.dump(regr,cachedir+filename) 
    return regr
        
        
if __name__ == "__main__":
    game_states_labeled=load_games_from_dir(datadir)
    print(game_states_labeled.info())
    
    x=game_states_labeled.drop('winner',1).values
    y=game_states_labeled['winner'].values
    
    #  train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split( x,
                                                                    y,
                                                                   test_size=0.20,
                                                                    random_state=42)
    print("sample train data size:{}".format(len(y_train)))
    

    #estimator=LinearRegression(n_jobs=-1)
    #poor prediction performance
   
    #estimator=KNeighborsRegressor(n_jobs = -1)
    #Knn was slow and didn't predict acuratly enough to bother with 
   
    #estimator=RandomForestRegressor(n_jobs =-1, random_state=42)
    #essentially the same as extra trees with slightly worse performance
    
    #estimator=svr()
    #absolutly horrible train/predict time, and no better performance than Linear
    

    print("\nstart ExtraTrees:")   
    estimator=ExtraTreesRegressor(n_jobs =-1)
    estimator=ExtraTreesClassifier(n_jobs =-1)
    
    print("default params of estimator",estimator)
    #use grid search to spot the best params
    #param=dict(n_estimators=[3,5,7,10,25,50,200,500], max_features=['auto','sqrt','log2'])
    param=dict(n_estimators=[3,10,50], max_features=['auto'])
    estimator=grid_search_wrapper(X_train,y_train,estimator,param,regr_name='ExtraTrees')
    print("post grid search params of est",estimator)
    #train the estimator
    start_time = time.time()
    estimator.fit(X_train,y_train)
    fit_time=time.time()-start_time
    print("fit time:{}s".format(round(fit_time, 3) ))
    #test on the validation set
    start_time = time.time()
    curr_predict=np.array(estimator.predict(X_validation)).copy()
    predict_time=time.time()-start_time
    print("predict time:{}s".format(round(predict_time, 3) ))    
    #track the run info
    MAE=np.mean(abs(curr_predict - y_validation))
    print("Mean abs error: {:.2f}".format(MAE))
    #retrain estimator with all data for final model
    estimator.fit(x,y)
    joblib.dump(estimator,"./trained_score_model.joblib")
    
    print(estimator.classes_)
    print(estimator.feature_importances_)

  
    #print(2-curr_predict[:5])
    #print(curr_predict[:5])
    #for cp in curr_predict:
        #if cp <=1 or cp>=2:
            #print(cp)
    #for i in range(10):
        #print(curr_predict[i], y_validation[i])
    #for x in X_validation[:10]:
        #print(to_string(list(x)))
        
        