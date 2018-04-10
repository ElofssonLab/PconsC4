
def compute_ranking(scores, min_separation=0):
    N = scores.shape[0]
    score_lst = []
    for i in range(N - min_separation):
        for j in range(i + min_separation, N):
            score_lst.append((i + 1, j + 1, scores[i, j]))
    score_lst.sort(key=lambda x: x[2], reverse=True)
    return score_lst
