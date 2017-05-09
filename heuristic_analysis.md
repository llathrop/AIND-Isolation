# Heuristic Analysis of AI Isolation

## Summary
 After implementing the MiniMax and AlphaBeta algorithms, we implemented 3 custom scoring heuristics which, when called, provide a score for the current game state. A higher score implies the current state is more likely to lead to a win for the current player. 

## Detail
 By way of example we were provided with several example score functions, the best performing of which(improved_score), simply returns a value equal to the difference between the playes remaining moves vs the opposing players remaining moves. The idea is that the player with more remaining moves has an advantage
 
 For our version of the scorer, we enhanced the improved_score version:
 
 * **custom_score_3:** We have added a metric about the game state, "empty_board", which contains the number of spaces left unblocked in the game. We divide the improved_score by this number. The idea behind this is that when the game begins, no particular move is very valuable, as we have plenty of room to move still<br>
 	``` (own_moves**2 - opp_moves**2)/empty_board	``` 
 * **custom_score_2:** Similar to the prior scorer, we have added another metric, "moves_left", which is the total number of moves left to all players. While simply substituting this in where empty_board was works, it provided no change in game outcome. instead we combined the two metrics, as a deeper indicator of how much game remains. In addition, we found that squaring the number of player moves led to an improved score.<br>
 ``` (own_moves**2 - opp_moves**2)/(moves_left+empty_board) ```
 * **custom_score:** In the final version of the scorer, we used the idea that during the early portion of the game, the distance from the center will pay an important role. To this end during the first half of the game, we modify the prior formula from custom score_2 by adding the "center_score" to the number of moves of the current player. "center_score" is generated using the code from the example scorers, and is the square of the distance from the center of the board to the position of the player. During the final half of the game, we procede as in custom_score_2.<br>
 ```((own_moves+center_score)**2 - opp_moves**2)/(moves_left+empty_board)```
 
 Additionally, I implemented a scoring model that outputs a probability that a specific game state leads to a win for a specific player, and then multiplied the improved score by this probability, etc. The model used ExtraTreesClassifier from scikit-learn. details are available in score_model in the git repository. This model proved to be to slow computationally and lost due reaching the timeout condition, and so is untested in actual play


## Conclusion
