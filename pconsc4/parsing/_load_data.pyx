#cython: cdivision=True, boundscheck=False, wraparound=True, embedsignature=True, language_level=3
from __future__ import division

import os
import glob
import math
import warnings
import subprocess

import tables
import tqdm

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log2, sin, cos

from Bio import pairwise2


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

def process_pdb(str pdb_file, str fasta_file):
    cdef str sequence = ''.join(line.strip() for line in open(fasta_file) if not line.startswith('>'))
    cdef float phi, psi
    cdef str this_line

    # Run DSSP
    p = subprocess.Popen('dssp ' + pdb_file, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    dssp_out = [line.decode() for line in stdout.splitlines()]

    # Skip header
    for i, line in enumerate(dssp_out):
        if line.startswith('  #'):
            dssp_out = dssp_out[i + 1:]
            break

    # Parse sequence from pdb and align:
    pdb_sequence = ''.join(line[13] for line in dssp_out if line[13] != '!')
    align = pairwise2.align.localxd(sequence, pdb_sequence, -10, -3, -3, -1)
    aligned_ref_seq, aligned_seq, _, _, _ = align[0]

    # Output data:
    seq_len = len(aligned_ref_seq)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] six_state = np.full((seq_len, 6), fill_value=1 / 6.,
                                                                        dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] dihedrals_sc = np.zeros((seq_len, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] dihedrals = np.zeros((seq_len, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] rsa = np.zeros((seq_len, 1), dtype=np.float32)
    valid = np.zeros(seq_len, dtype=np.bool)
    # TODO: consider TCO, Alpha, and Kappa angles

    # Auxiliary data
    ss_dict = dict(H=0, E=1, B=2, G=3, I=3, S=4, T=4)
    ss_dict[' '] = 5
    ss_dict['-'] = 5

    # From Wilke et al 2014
    max_acc = {'C': 167.0, 'T': 172.0, 'M': 224.0, 'P': 159.0, 'W': 285.0, 'I': 197.0, 'K': 236.0, 'D': 193.0,
               'N': 195.0, 'A': 129.0, 'F': 240.0, 'S': 155.0, 'V': 174.0, 'E': 223.0, 'Y': 263.0, 'R': 274.0,
               'H': 224.0, 'L': 201.0, 'Q': 225.0, 'G': 104.0}

    dssp_pointer = 0  # Which aa in the structure has been parsed
    for i in range(len(aligned_ref_seq)):
        if aligned_seq[i] != '-':
            this_line = dssp_out[dssp_pointer]
            aa = this_line[13]

            six_state[i, :] = 0.
            six_state[i, ss_dict[this_line[16]]] = 1.
            rsa[i, 0] = float(this_line[35:38]) / max_acc.get(aa, 200)

            phi = float(this_line[103:109]) * 0.0174533
            psi = float(this_line[109:115]) * 0.0174533
            dihedrals[i, 0] = phi
            dihedrals[i, 1] = psi
            dihedrals_sc[i, 0] = sin(phi)
            dihedrals_sc[i, 1] = cos(phi)
            dihedrals_sc[i, 2] = sin(psi)
            dihedrals_sc[i, 3] = cos(psi)
            
            valid[i] = 1
            dssp_pointer += 1

    three_state = np.zeros((six_state.shape[0], 3), dtype=np.float32)
    three_state[:, 0] = six_state[:, 0] + six_state[:, 3]
    three_state[:, 1] = six_state[:, 1] + six_state[:, 2]
    three_state[:, 2] = six_state[:, 4] + six_state[:, 5]
    dihedrals[dihedrals > np.pi] = 0
    return valid, three_state, six_state, rsa, dihedrals, dihedrals_sc


def main():
    h5_50 = tables.open_file('../data/data_jhE3.ur50.v2.h5', 'w', filters=tables.Filters(9, 'blosc:zstd'))
    h5_90 = tables.open_file('../data/data_jhE3.ur90.v2.h5', 'w', filters=tables.Filters(9, 'blosc:zstd'))

    pbar = tqdm.tqdm(glob.glob('/fat/pdbcull/E3/*.jhE3.ur50.fasta'), desc='UniRef50/90', leave=True)
    for fasta in pbar:
        try:
            group_name = 'p_' + fasta.split('/')[-1].split('.')[0]
            pdb = fasta.replace('E3/', 'pdb/').split('.jhE3.')[0] + '.pdb_fixed'
            os.system('vmtouch -qt {} &'.format(fasta))
            try:
                dssp_data = process_pdb(pdb, pdb.replace('.pdb_fixed', '.fa').replace('/pdb/', '/'))
            except:
                print('Problems with pdb', pdb)
                continue
            self_info, part_entr, seq = process_fasta(fasta)
            os.system('vmtouch -qe {} &'.format(fasta))

            group_50 = h5_50.create_group(h5_50.root, group_name)
            h5_50.create_carray(group_50, 'self_info', obj=self_info.astype(np.float32)[None, ...])
            h5_50.create_carray(group_50, 'part_entr', obj=part_entr.astype(np.float32)[None, ...])
            h5_50.create_carray(group_50, 'seq', obj=seq[None, ...])

            for name, arr in zip(('valid', 'ss_3', 'ss_6', 'rsa', 'dihedrals', 'dihedrals_sc'), dssp_data):
                h5_50.create_carray(group_50, name, obj=arr[None, :])
            h5_50.flush()

            self_info, part_entr, seq = process_fasta(fasta.replace('ur50.fasta', 'ur90.fasta'))
            group_90 = h5_90.create_group(h5_90.root, group_name)
            h5_90.create_carray(group_90, 'self_info', obj=self_info.astype(np.float32)[None, ...])
            h5_90.create_carray(group_90, 'part_entr', obj=part_entr.astype(np.float32)[None, ...])
            h5_90.create_carray(group_90, 'seq', obj=seq[None, ...])

            for name, arr in zip(('valid', 'ss_3', 'ss_6', 'rsa', 'dihedrals', 'dihedrals_sc'), dssp_data):
                h5_90.create_carray(group_90, name, obj=arr[None, :])
            h5_90.flush()
        except Exception as e:
              print('Processing', pdb, e, 'was raised.')
              print()
    h5_50.close()
    h5_90.close()
