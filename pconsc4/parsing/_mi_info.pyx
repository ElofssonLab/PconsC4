#cython: cdivision=True, boundscheck=False, wraparound=False, embedsignature=True, language_level=3
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log, sqrt

ctypedef np.int8_t int8

cdef void marginal(const int8[::1] column, long[::1] marginal) nogil:
    cdef Py_ssize_t i

    marginal[:] = 0
    for i in range(column.shape[0]):
        marginal[column[i]] += 1


cdef float entropy(const long[::1] marginal_labels) nogil:
    """Calculates the entropy for a labeling."""
    cdef float[::1] pi
    cdef Py_ssize_t i
    cdef float entr = 0.0, total = 0.0

    cdef float epsilon = 1e-7

    for i in range(marginal_labels.shape[0]):
        total += marginal_labels[i]

    for i in range(marginal_labels.shape[0]):
        if marginal_labels[i] > 0:
            entr -= marginal_labels[i] / total * (log(marginal_labels[i]) - log(total))
    entr = max(entr, epsilon)
    return entr


cdef void compute_contingency(const int8[::1] labels_true, const int8[::1] labels_pred, long[:, ::1] contingency,
                              long[::1] marginal_i, long[::1] marginal_j) nogil:
    cdef Py_ssize_t i, ix_i, ix_j, N = labels_pred.shape[0]

    contingency[:] = 0
    marginal_i[:] = 0
    marginal_j[:] = 0

    for i in range(N):
        ix_i = labels_pred[i]
        ix_j = labels_true[i]
        contingency[ix_i, ix_j] += 1
        marginal_i[ix_i] += 1
        marginal_j[ix_j] += 1


cdef np.float32_t compute_mi(const long[:, ::1] contingency, const long[::1] marginal_i, const long[::1] marginal_j) nogil:
    cdef Py_ssize_t i, j, N = contingency.shape[0]
    cdef float contingency_sum = 0, mi = 0, p_xy

    for i in range(N):
        contingency_sum += marginal_i[i]

    for i in range(N):
        for j in range(N):
            p_xy = contingency[i, j] / contingency_sum
            if p_xy != 0:
                mi += p_xy * (log(p_xy) - log(marginal_i[i] * marginal_j[j] / (contingency_sum * contingency_sum)))
    return mi



def load_aln(aln):
    cdef str line
    cdef list parsed
    mapping = {'-': 0, 'A': 1, 'B': 3, 'C': 2, 'D': 5, 'E': 4, 'F': 7, 'G': 6, 'H': 9,
               'I': 8, 'K': 10, 'L': 12, 'M': 11, 'N': 13, 'P': 15, 'Q': 14, 'R': 17,
               'S': 16, 'T': 18, 'V': 20, 'W': 19, 'X': 22, 'Y': 21}
    parsed = []
    for line in open(aln):
        line = line.strip()
        parsed.append([mapping.get(ch, 22) for ch in line])

    return np.array(parsed, dtype=np.int8, order='F').T


cdef float[:, ::1] apc_correction(const float[:, ::1] matrix, N):
    cdef float corr, mean
    cdef float[::1] mean_x, mean_y
    #cdef float[:, ::1] corrected
    cdef Py_ssize_t i, j

    mean_x = np.mean(matrix, axis=0)
    mean_y = np.mean(matrix, axis=1)
    mean = np.mean(matrix)
    corrected = matrix.copy()
    for i in range(N):
        for j in range(i, N):
            corr = mean_x[j] * mean_y[i] / mean
            corrected[i, j] -= corr
            corrected[j, i] -= corr

    return corrected


cdef void _compute_mi_scores(int8[:, ::1] alignment, long[::1] marginal_x, long[::1] marginal_y, long[:, ::1] contingency,
                             float[::1] entropies,
                             float[:, ::1] cross_h,
                             float[:, ::1] mi,
                             float[:, ::1] nmi, int N) nogil:
    cdef Py_ssize_t i, j
    cdef np.float32_t _h, _mi
    cdef int8[::1] col_a, col_b

    for i in range(N):
        col_a = alignment[i, :]
        marginal(col_a, marginal_x)
        entropies[i] = entropy(marginal_x)
    for i in range(N):
        col_a = alignment[i, :]
        for j in range(i, N):
            col_b = alignment[j, :]

            compute_contingency(col_a, col_b, contingency, marginal_x, marginal_y)
            _mi = compute_mi(contingency, marginal_x, marginal_y)

            _h = entropies[i] + entropies[j] - _mi
            cross_h[i, j] = _h
            cross_h[j, i] = _h
            mi[i, j] = mi[j, i] = _mi
            nmi[i, j] = nmi[j, i] = _mi / sqrt(entropies[i] * entropies[j])


def compute_mi_scores(int8[:, ::1] alignment):
    cdef Py_ssize_t i, j, N = alignment.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] mi, nmi, cross_h
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] entropies = np.zeros(N, dtype=np.float32)
    #cdef float[:, ::1] mi, nmi, cross_h
    #cdef float[::1] entropies

    cdef long contingency_[23][23]
    cdef long marginal_x_[23]
    cdef long marginal_y_[23]

    cdef long[:, ::1] contingency = contingency_
    cdef long[::1] marginal_x = marginal_x_, marginal_y = marginal_y_

    cdef dict results

    mi = np.zeros((N, N), dtype=np.float32)
    nmi = np.zeros((N, N), dtype=np.float32)
    cross_h = np.zeros((N, N), dtype=np.float32)


    _compute_mi_scores(alignment, marginal_x, marginal_y,contingency, entropies, cross_h, mi, nmi, N)

    results = dict(mi=mi, nmi=nmi, cross_h=cross_h, entropies=entropies,
                   mi_corr=np.array(apc_correction(mi, N)), nmi_corr=np.array(apc_correction(nmi, N)))
    return results
