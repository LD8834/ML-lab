MAX, MIN = 1000, -1000

def alphabeta_minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta, path, prune_count):
    if depth == 3:
        return values[nodeIndex], path, prune_count

    if maximizingPlayer:
        best = MIN
        best_path = []
        for i in range(0, 2):
            val, new_path, prune_count = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta, path + [nodeIndex * 2 + i], prune_count)
            if val > best:
                best = val
                best_path = new_path
            alpha = max(alpha, best)
            if beta <= alpha:
                prune_count += 1
                break
        return best, best_path, prune_count

    else:
        best = MAX
        best_path = []
        for i in range(0, 2):
            val, new_path, prune_count = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, path + [nodeIndex * 2 + i], prune_count)
            if val < best:
                best = val
                best_path = new_path
            beta = min(beta, best)
            if beta <= alpha:
                prune_count += 1
                break
        return best, best_path, prune_count

values = [3, 5, 6, 9, 1, 2, 0, -1]
optimal_value, optimal_path, prune_count = alphabeta_minimax(0, 0, True, values, MIN, MAX, [], 0)
print("The optimal value is:", optimal_value)
print("The optimal path is:", optimal_path)
print("Number of pruned nodes:", prune_count)
