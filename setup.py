from __future__ import division
import os
from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize

# Check wheter we have pyx or c files:
base_path = os.path.dirname(os.path.abspath(__file__))

# A sdist will have C files, use those:
if os.path.exists(os.path.join(base_path, 'pconsc4/parsing/_load_data.c')):
    use_cython = False
else:
    # It appears we are on git, go ahead and cythonice everything
    use_cython = True

flags = "-O2 -march=native -pipe -mtune=native".split()

if use_cython:
    extensions = [
        Extension(
            'pconsc4.parsing._load_data', ['pconsc4/parsing/_load_data.pyx'],
            extra_compile_args=flags,
            extra_link_args=flags),
        Extension(
            'pconsc4.parsing._mi_info', ['pconsc4/parsing/_mi_info.pyx'],
            extra_compile_args=flags,
            extra_link_args=flags)
    ]
else:
    extensions = [
        Extension(
            'pconsc4.parsing._load_data', ['pconsc4/parsing/_load_data.c'],
            extra_compile_args=flags,
            extra_link_args=flags),
        Extension(
            'pconsc4.parsing._mi_info', ['pconsc4/parsing/_mi_info.c'],
            extra_compile_args=flags,
            extra_link_args=flags)
    ]

setup(
    name='pconsc4',
    version='0.1',
    description='',
    url='https://github.com/ElofssonLab/PconsC4',
    author='Mirco Michel and David Menéndez Hurtado',
    author_email='davidmenhur@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    package_data={
        'pconsc4.models': [
            'pconsc4/models/pconsc4_unet_weights.h5',
            'pconsc4/models/ss_pred_resnet_elu_nolr_dropout01_l26_large_v3_saved_model.h5'
        ]
    },
    include_package_data=True,
    ext_modules=cythonize(
        ['pconsc4/parsing/_load_data.pyx', 'pconsc4/parsing/_mi_info.pyx']),
    requires=['numpy', 'Cython', 'scipy', 'keras', 'gaussdca', 'h5py'],
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: '
        'GNU General Public License v3 (GPLv3)'
    ],
    zip_safe=False)

