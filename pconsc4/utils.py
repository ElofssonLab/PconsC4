def compute_ranking(scores, min_separation=0):
    N = scores.shape[0]
    score_lst = []
    for i in range(N - min_separation):
        for j in range(i + min_separation, N):
            score_lst.append((i + 1, j + 1, scores[i, j]))
    score_lst.sort(key=lambda x: x[2], reverse=True)
    return score_lst


def format_contacts_casp(scores, sequence, properties=None, min_sep=0, full_precision=True):
    scores = scores.squeeze()

    # Set default values:
    if properties is None:
        properties = dict()

    default_properties = dict(target_name='TARGET',
                              author='',
                              group='PconsC4',
                              developer='Mirco Michel, David Men\\\'endez Hurtado and Arne Elofsson',
                              method='PconsC4')
    known_keys = list(default_properties.keys())

    default_properties.update(properties)
    # Header:
    header = '\n'.join(("PFRMAT RR",
                        "TARGET {target_name}",
                        "AUTHOR {author}",
                        "METHOD {method}",
                        "REMARK GROUP {group}",
                        "REMARK DEVELOPER {developer}\n")).format(**default_properties)

    # Any other information passed in properties will be added as a remark
    for k in known_keys:
        default_properties.pop(k)

    content = [header]
    for key, value in default_properties.items():
        content.append('REMARK {} {}\n'.format(key, value))
    content.append("MODEL 1\n")

    # Save the sequence, wrapped at 50 chars for some reason.
    n = 50
    content.extend(sequence[i:i + n] + '\n' for i in range(0, len(sequence), n))

    if full_precision:
        line = '{i} {j} 0 8 {score}\n'
    else:
        line = '{i} {j} 0 8 {score:.8f}\n'

    data = []
    for i in range(len(sequence) - 1):
        for j in range(i + 1, len(sequence)):
            if abs(i - j) > min_sep:
                data.append((i, j, scores[i, j]))

    data.sort(key=lambda x: x[2], reverse=True)
    for i, j, sc in data:
        content.append(line.format(i=i + 1, j=j + 1, score=sc))

    content.append('END\n')
    return ''.join(content)


def format_ss3(ss3_pred):
    codes = {0: 'H', 1: 'E', 2: 'C'}
    return ''.join(codes[x] for x in ss3_pred.squeeze().argmax(axis=1))


def format_contacts_cameo(scores, sequence, sequence_id='A', min_sep=0, full_precision=True):
    if full_precision:
        template = 's{}.{}\ts1.{}\t0\t8\t{}\n'
    else:
        template = 's{}.{}\ts1.{}\t0\t8\t{:.4f}\n'

    scores = scores.squeeze()

    # Convert sequence to list with numbered residues
    residues = [char + str(i + 1) for i, char in enumerate(sequence)]

    data = []
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if abs(i - j) > min_sep:
                data.append((i, j, scores[i, j]))

    # Sort list, starting with the largest probability
    data.sort(key=lambda x: x[2], reverse=True)

    lines = []
    for i in range(len(data)):
        a = data[i][0]
        b = data[i][1]
        prob = data[i][2]
        current_line = template.format(sequence_id, residues[a], residues[b], prob)
        lines.append(current_line)
    lines.append('\n')
    return '\n'.join(lines)
