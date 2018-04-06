from __future__ import division
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

setup(name='pconsc4', version='0.1',
      description='',
      url='',
      author='Mirco Michel and David Menéndez Hurtado',
      author_email='davidmenhur@gmail.com',
      license='GPLv3',
      packages=find_packages(),
      package_data={'pconsc4.models': ['pconsc4/models/pconsc4_unet_weights.h5',
                                       'pconsc4/models/ss_pred_resnet_elu_nolr_dropout01_l26_large_v3_saved_model.h5']},
      include_package_data=True,
      ext_modules=cythonize(['pconsc4/parsing/_load_data.pyx', 'pconsc4/parsing/_mi_info.pyx']),
      requires=['numpy', 'Cython', 'scipy', 'keras', 'gaussdca'],
      classifiers=['Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: '
                   'GNU General Public License v3 (GPLv3)'],
      zip_safe=False)
