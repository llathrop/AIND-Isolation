"""This file contains a collection of player classes for comparison with your
own agent and example heuristic functions.

    ************************************************************************
    ***********  YOU DO NOT NEED TO MODIFY ANYTHING IN THIS FILE  **********
    ************************************************************************
"""

from random import randint


def null_score(game, player):
    """This heuristic presumes no knowledge for non-terminal states, and
    returns the same uninformative value for all other states.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 0.


def open_move_score(game, player):
    """The basic evaluation function described in lecture that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width, game.height
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


class RandomPlayer():
    """Player that chooses a move randomly."""

    def get_move(self, game, time_left):
        """Randomly select a move from the available legal moves.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            A randomly selected legal move; may return (-1, -1) if there are
            no available legal moves.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        return legal_moves[randint(0, len(legal_moves) - 1)]


class GreedyPlayer():
    """Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """

    def __init__(self, score_fn=open_move_score):
        self.score = score_fn

    def get_move(self, game, time_left):
        """Select the move from the available legal moves with the highest
        heuristic score.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            The move in the legal moves list with the highest heuristic score
            for the current game state; may return (-1, -1) if there are no
            legal moves.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        _, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
        return move


class HumanPlayer():
    """Player that chooses a move according to user's input."""

    def get_move(self, game, time_left):
        """
        Select a move from the available legal moves based on user input at the
        terminal.

        **********************************************************************
        NOTE: If testing with this player, remember to disable move timeout in
              the call to `Board.play()`.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            The move in the legal moves list selected by the user through the
            terminal prompt; automatically return (-1, -1) if there are no
            legal moves
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        print(game.to_string()) #display the board for the human player
        print(('\t'.join(['[%d] %s' % (i, str(move)) for i, move in enumerate(legal_moves)])))

        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < len(legal_moves)

                if not valid_choice:
                    print('Illegal move! Try again.')

            except ValueError:
                print('Invalid index! Try again.')

        return legal_moves[index]
    
   
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
def save_game(state_hist,winner,game_file):
    """ Save the state of the game and the winner 
    preserve the game to a file for later analysis
    each row in the file is the game state.
    row has format [gamestate],player_num_of_winner
    """
    import pickle
    try:
        with open(game_file, 'wb') as f:
            pickle.dump((state_hist,winner),f)
    except:
        return "failed"
    return "success"

def load_game(game_file):
    """ load the state of the game and the winner 
    load the game from a file for later analysis
    each row in the file is the game state.
    row has format [gamestate],player_num_of_winner
    """
    import pickle
    try:
        with open(game_file, 'rb') as f:
            state=pickle.load(f)
    except:
        return "failed",_
    return "success",state


from game_agent import (MinimaxPlayer, AlphaBetaPlayer, custom_score,
                        custom_score_2, custom_score_3)

if __name__ == "__main__":
    from isolation import Board

    # create an isolation board (by default 7x7)
    player1 = RandomPlayer()
    player2 = GreedyPlayer()
    player1 = AlphaBetaPlayer()
    player2 = MinimaxPlayer()
    game = Board(player1, player2)

    # place player 1 on the board at row 2, column 3, then place player 2 on
    # the board at row 0, column 5; display the resulting board state.  Note
    # that the .apply_move() method changes the calling object in-place.
    game.apply_move((2, 3))
    game.apply_move((0, 5))
    print(game.to_string())

    # players take turns moving on the board, so player1 should be next to move
    assert(player1 == game.active_player)

    # get a list of the legal moves available to the active player
    print(game.get_legal_moves())

    # get a successor of the current state by making a copy of the board and
    # applying a move. Notice that this does NOT change the calling object
    # (unlike .apply_move()).
    new_game = game.forecast_move((1, 1))
    assert(new_game.to_string() != game.to_string())
    print("\nOld state:\n{}".format(game.to_string()))
    print("\nNew state:\n{}".format(new_game.to_string()))

    # play the remainder of the game automatically -- outcome can be "illegal
    # move", "timeout", or "forfeit"
    winner, history, outcome,state_hist = game.play()
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    print(game.to_string())
    print("Move history:\n{!s}".format(history))
    
    #game history
    player1_moves=[]
    player1_states=[]
    player2_moves=[]
    player2_states=[]
    for i in range(len(history)):
        if i%2 ==0:
            player1_moves.append(history[i])
            player1_states.append(state_hist[i])
        else:
            player2_moves.append(history[i])
            player2_states.append(state_hist[i])
            
    print(len(player1_moves),len(player2_moves))
    #print the players next move and the current board
    if len(player1_moves)==len(player2_moves):
        for i in range(len(player1_moves)):
            print('player1:{}\n{} \n player2:{}\n{}'.format(player1_moves[i],player1_states[i],player2_moves[i],player2_states[i]))
    else:
        if len(player1_moves)>len(player2_moves):
            for i in range(len(player2_moves)):
                print('player1:{}\n{} \nplayer2:{}\n{}'.format(player1_moves[i],to_string(player1_states[i]),player2_moves[i],to_string(player2_states[i])))
            print(' player1:{}\n{}'.format(player1_moves[i+1],to_string(player1_states[i+1])))

        else:
            for i in range(len(player1_moves)):
                print('player1:{}\n{}\n player2:{}\n{}'.format(player1_moves[i],player1_states[i],player2_moves[i],player2_states[i]))
            print('       player2:{}{}'.format(player2_moves[i+1],player2_states[i]))

    print("final state:\n",game.to_string())
    if winner == game._player_1:
        result, state=save_game(state_hist,1,"sample_players.pckl")
    else:
        result, state=save_game(state_hist,2,"sample_players.pckl")
    
    result,state=load_game("sample_players.pckl")
    
    print(result,"test:\n")