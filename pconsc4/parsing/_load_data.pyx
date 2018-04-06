#cython: cdivision=True, boundscheck=False, wraparound=True, embedsignature=True, language_level=3
from __future__ import division

import os
import glob
import math
import warnings
import subprocess

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log2, sin, cos


ctypedef np.int8_t int8

def load_aln(str aln):
    cdef str line
    cdef list parsed
    cdef dict mapping = {'-': 0, 'A': 1, 'B': 3, 'C': 2, 'D': 5, 'E': 4, 'F': 7,
                         'G': 6,'H': 9, 'I': 8, 'K': 10, 'L': 12, 'M': 11,
                         'N': 13, 'P': 15,'Q': 14, 'R': 17, 'S': 16, 'T': 18,
                         'V': 20, 'W': 19, 'Y': 21,
                         'O': 10, 'U': 2, 'Z': 4, 'X': 22}
    parsed = []
    for line in open(aln):
        line = line.strip()
        parsed.append([mapping.get(ch, 22) for ch in line])

    return np.array(parsed, dtype=np.int8, order='F').T

def load_fasta(str fasta):
    cdef str line
    cdef list parsed
    cdef dict mapping = {'-': 0, 'A': 1, 'B': 3, 'C': 2, 'D': 5, 'E': 4, 'F': 7,
                         'G': 6,'H': 9, 'I': 8, 'K': 10, 'L': 12, 'M': 11,
                         'N': 13, 'P': 15,'Q': 14, 'R': 17, 'S': 16, 'T': 18,
                         'V': 20, 'W': 19, 'Y': 21,
                         'O': 10, 'U': 2, 'Z': 4, 'X': 22}
    parsed = []
    for line in open(fasta):
        if line.startswith('>'):
            continue
        line = line.strip()
        parsed.append([mapping.get(ch, 22) for ch in line])

    return np.array(parsed, dtype=np.int8, order='F').T

def load_a3m(str fasta):
    cdef str line
    cdef list parsed
    cdef dict mapping = {'-': 0, 'A': 1, 'B': 3, 'C': 2, 'D': 5, 'E': 4, 'F': 7,
                         'G': 6,'H': 9, 'I': 8, 'K': 10, 'L': 12, 'M': 11,
                         'N': 13, 'P': 15,'Q': 14, 'R': 17, 'S': 16, 'T': 18,
                         'V': 20, 'W': 19, 'Y': 21,
                         'O': 10, 'U': 2, 'Z': 4, 'X': 22}
    parsed = []
    for line in open(fasta):
        if line.startswith('>'):
            continue
        line = line.strip()
        parsed.append([mapping.get(ch, 22) for ch in line if not ch.islower()])

    return np.array(parsed, dtype=np.int8, order='F').T

@cython.wraparound(False)
cdef void compute_marginal(const int8[::1] column, double[::1] marginal_, int N) nogil:
    cdef Py_ssize_t i
    cdef double total = N + 23

    marginal_[:] = 0
    for i in range(N):
        marginal_[column[i]] += 1

    for i in range(23):
        marginal_[i] += 1 # Add a pseudocount to avoid missing cases
        marginal_[i] /= total

@cython.wraparound(False)
cdef void _compute_values(int8[:, ::1] alignment, double[::1] marginal,
                          double[:, ::1] self_info, # -log(p)
                          double[:, ::1] part_entr, # -p log(p)
                          int nb_sequences, int nb_columns) nogil:
    cdef Py_ssize_t i, j, counter=0
    cdef double b
    cdef int8[::1] col

    cdef double *backgrounds = [0.0, 0.08419317947262096, 0.014738128673144382, 5.436173540314418e-07, 0.06197693918988444, 0.05459461427571707, 0.06674515639448204, 0.038666461097043095, 0.05453584555390309, 0.022640692546158938, 0.052128474295457077, 0.021836718625738008, 0.09559465355790236, 0.042046100985207224, 0.040134181479253114, 0.05077784852151325, 0.07641242434788177, 0.058285168924796175, 0.05738649269469764, 0.012712588510636218, 0.06437953006998008, 0.02941669643528118, 0.0007975607313478385]

    #Estimate background probability of gap
    for i in range(nb_columns):
        for j in range(nb_sequences):
            if alignment[i, j] == 0:
                counter += 1
    backgrounds[0] = (counter + 1.) / (nb_columns * nb_sequences)

    for i in range(nb_columns):
        col= alignment[i, :]
        compute_marginal(col, marginal, nb_sequences)

        for j in range(23):
            b = backgrounds[j]
            self_info[i, j] = -log2(marginal[j] / b)
            part_entr[i, j] = -marginal[j] * log2(marginal[j] / b)

@cython.wraparound(False)
cdef compute_values(int8[:, ::1] alignment):
    cdef Py_ssize_t i, j, number_columns, number_sequences
    number_columns = alignment.shape[0]
    number_sequences = alignment.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] self_info
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] part_entr

    self_info = np.empty((number_columns, 23), dtype=np.float64)
    part_entr = np.empty((number_columns, 23), dtype=np.float64)

    cdef double marginal_[23]
    cdef double[::1] marginal = marginal_

    _compute_values(alignment, marginal, self_info, part_entr,
                    number_sequences, number_columns)

    return self_info, part_entr

@cython.wraparound(False)
def process_fasta(str fasta):
    cdef Py_ssize_t i, number_columns, number_sequences

    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] seq
    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] alignment
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] self_info
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] part_entr

    alignment = load_fasta(fasta)

    number_columns = alignment.shape[0]
    number_sequences = alignment.shape[1]

    seq = np.zeros((number_columns, 22), dtype=np.int8)

    for i in range(number_columns):
        seq[i, alignment[i, 0] - 1] = 1

    self_info, part_entr = compute_values(alignment)

    return self_info, part_entr, seq

@cython.wraparound(False)
def process_aln(str fasta):
    cdef Py_ssize_t i, number_columns, number_sequences

    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] seq
    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] alignment
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] self_info
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] part_entr

    alignment = load_aln(fasta)

    number_columns = alignment.shape[0]
    number_sequences = alignment.shape[1]

    seq = np.zeros((number_columns, 22), dtype=np.int8)

    for i in range(number_columns):
        seq[i, alignment[i, 0] - 1] = 1

    self_info, part_entr = compute_values(alignment)

    return self_info, part_entr, seq

@cython.wraparound(False)
def process_a3m(str fasta):
    cdef Py_ssize_t i, number_columns, number_sequences

    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] seq
    cdef np.ndarray[np.int8_t, ndim=2, mode='c'] alignment
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] self_info
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] part_entr

    alignment = load_a3m(fasta)

    number_columns = alignment.shape[0]
    number_sequences = alignment.shape[1]

    seq = np.zeros((number_columns, 22), dtype=np.int8)

    for i in range(number_columns):
        seq[i, alignment[i, 0] - 1] = 1

    self_info, part_entr = compute_values(alignment)

    return self_info, part_entr, seq
