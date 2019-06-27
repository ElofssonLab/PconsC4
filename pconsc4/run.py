from __future__ import unicode_literals
import os

import numpy as np
import gaussdca

from .parsing._load_data import process_a3m, load_a3m
from .parsing._mi_info import compute_mi_scores


def _pad(x, depth=4):
    """ Add padding for unet of given depth """
    divisor = np.power(2, depth)
    remainder = x.shape[0] % divisor

    # no padding because already of even shape
    if remainder == 0:
        return x
    # add zero rows after 1D feature
    elif len(x.shape) == 2:
        return np.pad(x, [(0, divisor - remainder), (0, 0)], "constant")
    # add zero columns and rows after 2D feature
    elif len(x.shape) == 3:
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder),
                          (0, 0)], "constant")


def _generate_features(fname, verbose=0):
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr',
                'self_info']

    if not os.path.isfile(fname):
        raise IOError("Alignment file does not exist.")

    if verbose > 1:
        print('Extracting column statistics')
    self_info, part_entr, seq = process_a3m(fname)
    seq_dict = {'seq': seq, 'part_entr': part_entr, 'self_info': self_info}
    original_length = seq.shape[-2]

    if verbose > 1:
        print('Computing mutual information')
    a3m_ali = load_a3m(fname)
    mi_dict = compute_mi_scores(a3m_ali)

    if verbose > 1:
        print('Running GaussDCA')
    gdca_dict = gaussdca.run(fname)

    feat_dict = {}
    for feat in feat_lst:
        if feat == 'gdca':
            x_i = gdca_dict['gdca_corr']
            x_i = x_i[..., None]
        elif feat in ['cross_h', 'nmi_corr', 'mi_corr']:
            x_i = mi_dict[feat]
            x_i = x_i[..., None]
        elif feat in ['seq', 'part_entr', 'self_info']:
            x_i = seq_dict[feat]
        else:
            raise ValueError('Unkown feature {}'.format(feat))
        L = x_i.shape[0]
        x_i = _pad(x_i)
        feat_dict[feat] = x_i[None, ...]

    mask = np.ones((L, L))
    mask = mask[..., None]  # reshape from (L,L) to (L,L,1)
    mask = _pad(mask)
    feat_dict['mask'] = mask[None, ...]  # reshape from (L,L,1) to (1,L,L,1)

    return feat_dict, original_length, gdca_dict['eff_seq']


def _symmetrize(matrix, L):
    matrix = np.squeeze(matrix[:, :L, :L, :])  # Remove padding
    matrix = (matrix + matrix.T) / 2.  # Symmetrize
    np.fill_diagonal(matrix, 0)
    return matrix


def _predict_contacts(model, feat_dict, L):
    result_lst = [_symmetrize(x, L) for x in model.predict_on_batch(feat_dict)]

    return dict(cmap=result_lst[2], s_score=result_lst[0], cmap_6=result_lst[1],
                cmap_10=result_lst[3], features=feat_dict)


def _predict_ss(model, feat_dict, L):
    feat_list = [feat_dict[k] for k in ('seq', 'self_info', 'part_entr')]
    result_lst = [x[:, :L, :] for x in model.predict_on_batch(feat_list)]
    names = ('ss3', 'ss6', 'rsa', 'dihedrals')

    return dict(zip(names, result_lst), features=feat_dict)


def predict(model, alignment, verbose=0):
    return predict_contacts(model, alignment, verbose)


def predict_contacts(model, alignment, verbose=0):
    feat_dict, L, meff = _generate_features(alignment, verbose)
    if verbose:
        print('Features generated')
        print('Predicting')

    results = _predict_contacts(model.contact_model, feat_dict, L)
    results['eff_seq'] = meff
    return results


def predict_ss(model, alignment, verbose=0):
    feat_dict, L, meff = _generate_features(alignment, verbose)
    if verbose:
        print('Features generated')
        print('Predicting')
    results = _predict_ss(model.ss_model, feat_dict, L)
    results['eff_seq'] = meff
    return results


def predict_all(model, alignment, verbose=0):
    feat_dict, L, meff = _generate_features(alignment, verbose)
    if verbose:
        print('Features generated')
        print('Predicting')

    results = dict(features=feat_dict)
    results['contacts'] = _predict_contacts(model.contact_model, feat_dict, L)
    results['ss'] = _predict_ss(model.ss_model, feat_dict, L)
    results['eff_seq'] = meff
    return results
