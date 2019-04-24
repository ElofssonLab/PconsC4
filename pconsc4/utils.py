def compute_ranking(scores, min_separation=0):
    N = scores.shape[0]
    score_lst = []
    for i in range(N - min_separation):
        for j in range(i + min_separation, N):
            score_lst.append((i + 1, j + 1, scores[i, j]))
    score_lst.sort(key=lambda x: x[2], reverse=True)
    return score_lst


def format_contacts_cameo(scores, sequence, seqidx, filename_out, min_sep=0):
    scores = scores.squeeze()

    # Read sequence
    f = open(sequence)
    txt = []
    for line in f:
        txt.append(line)
    f.close()
    seq = txt[1]
    seq = seq.strip()

    # Convert sequence to list with numbered residues
    i = 0
    residues = list()

    while i < len(seq):
        elem = seq[i] + str(i + 1)
        residues.append(elem)
        i += 1

    data = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if abs(i - j) > min_sep:
                data.append((i, j, scores[i, j]))

    # Sort list, starting with the largest probability
    data.sort(key=lambda x: x[2], reverse=True)

    # Write lines
    line_list = list()
    for i in range(len(data)):
        a = data[i][0]
        b = data[i][1]
        prob = data[i][2]
        current_line = "s%s.%s\ts1.%s\t0\t8\t%.4f" % (seqidx, residues[a], residues[b], prob)
        line_list.append(current_line)

    # Write lines into text file
    txt = open(filename_out, 'a')
    txt.write('\n'.join(line_list))
    txt.close()


def format_ss3(ss3_pred):
    codes = {0: 'H', 1: 'E', 2: 'C'}
    return ''.join(codes[x] for x in ss3_pred.squeeze().argmax(axis=1))
