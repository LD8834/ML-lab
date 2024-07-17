4b) 
MAX, MIN = 1000, -1000

def alphabeta_minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta, path):

    if depth == 3:
        return values[nodeIndex], path

    if maximizingPlayer:
        best = MIN
        best_path = []
        for i in range(0, 2):
            val, new_path = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta, path + [nodeIndex * 2 + i])
            if val > best:
                best = val
                best_path = new_path
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best, best_path

    else:
        best = MAX
        best_path = []
        for i in range(0, 2):
            val, new_path = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, path + [nodeIndex * 2 + i])
            if val < best:
                best = val
                best_path = new_path
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_path
values = [3, 5, 6, 9, 1, 2, 0, -1]
optimal_value, optimal_path = alphabeta_minimax(0, 0, True, values, MIN, MAX, [])
print("The optimal value is:", optimal_value)
print("The optimal path is:", optimal_path)


4a)
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('./ToyotaCorolla.csv')

plt.boxplot([data["Price"],data["HP"],data["KM"]])

plt.xticks([1,2,3],["Price","HP","KM"])

plt.show()
