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
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder), (0, 0)], "constant")


def _generate_features(fname):
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']

    if not os.path.isfile(fname):
        raise IOError("Alignment file does not exist.")

    # self_info, part_entr, seq = process_a3m(unicode(fname, encoding="utf-8"))
    self_info, part_entr, seq = process_a3m(fname)
    seq_dict = {'seq': seq, 'part_entr': part_entr, 'self_info': self_info}

    a3m_ali = load_a3m(fname)
    mi_dict = compute_mi_scores(a3m_ali)

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

    return feat_dict, L


def _predict(model, feat_dict, L):
    result_lst = model.predict_on_batch(feat_dict)
    cmap = result_lst[2][:, :L, :L, :]  # remove padding
    cmap = cmap.reshape((L, L))
    cmap = (cmap + cmap.T) / 2.  # make it symmetric by averaging over triangles
    sscoremap = result_lst[2][:, :L, :L, :]
    sscoremap = sscoremap.reshape((L, L))
    sscoremap = (sscoremap + sscoremap.T) / 2.

    return dict(cmap=cmap, s_score=sscoremap)



def predict(model, alignment):
    feat_dict, L = _generate_features(alignment)
    return _predict(model, feat_dict, L)
