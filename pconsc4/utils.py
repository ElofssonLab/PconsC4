def compute_ranking(scores, min_separation=0):
    N = scores.shape[0]
    score_lst = []
    for i in range(N - min_separation):
        for j in range(i + min_separation, N):
            score_lst.append((i + 1, j + 1, scores[i, j]))
    score_lst.sort(key=lambda x: x[2], reverse=True)
    return score_lst


def format_contacts_casp(scores, sequence, properties=None):
    # Set default values:
    if properties is None:
        properties = dict()

    default_properties = dict(target_name='TARGET', res_num=len(sequence),
                              author='5229-7541-3942', group='Elofsson',
                              developer='Mirco Michel, David Menendez-Hurtado and Arne Elofsson',
                              method='PconsC4')

    default_properties.update(properties)
    # Header:
    header = '''PFRMAT RR
TARGET {target_name}, , {res_num} residues
AUTHOR {author}
REMARK GROUP {group}
REMARK DEVELOPER {developer}
METHOD {method}
MODEL  1
'''.format(**default_properties)

    content = [header]
    line = '{i} {j} 0 8 {score}\n'
    for i in range(len(sequence) - 1):
        for j in range(i + 1, len(sequence)):
            content.append(line.format(i=i + 1, j=j + 1, scores=scores[i, j]))

    return ''.join(content)


def format_ss3(ss3_pred):
    codes = {0: 'H', 1: 'S', 2: 'C'}
    return ''.join(codes[x] for x in ss3_pred.argmax(axis=1))
