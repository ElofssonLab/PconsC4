# PconsC4:
Fast, accurate, and hassle-free contact prediction.

## Installation instructions:

Using conda:

    conda env create -f pconsc4_env.yml
    conda activate pconsc4_env
    pip install pconsc4
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"
    conda deactivate



Using pip:

    pip3 install numpy Cython pythran &&
    pip3 install pconsc4

NB: the trained model is a bit over Github's limit, so they cannot be checked in the repo.
If you need them, you can grab the trained models from the [releases tab.](https://github.com/ElofssonLab/PconsC4/releases)

You will also need a deep learning backend compatible with Keras. We recommend Tensorflow:

    pip3 install -U tensorflow

## Versions

These versions are known to work

keras==2.2.4
2.0< tensorflow >=1.12.
pythran 0.9.5 

Later versions (such as tensorflow 2) are known to not work.


## Usage instructions

Inside Python:

    import pconsc4

    model = pconsc4.get_pconsc4()

    pred_1 = pconsc4.predict(model, 'path/to/alignment1')
    pred_2 = pconsc4.predict(model, 'path/to/alignment2')
    
    # Show pred_1 on the screen:
    
    import matplotlib.pyplot as plt 
    plt.imshow(pred_1['cmap'])
    plt.show()


The program accepts alignments in .fasta, .a3m, or .aln, without line wrapping.
The query sequence must be the first line, and it cannot contain gaps.

We also provide a function to format the output in CASP format: 

    # Save in CASP format:
    from pconsc4.utils import format_contacts_casp
    print(format_contacts_casp(pred_2['cmap'], seq_2, min_sep=5))

and Cameo:

    # Save in Cameo format:
    from pconsc4.utils import format_contacts_cameo
    print(format_contacts_cameo(pred_2['cmap'], seq_2, min_sep=5))
   
## Troubleshooting:

* #### pyGaussDCA fails with "template errors":
  Update your compiler. GCC 5 or higher is known to work.
* #### I get a Segmentation Fault when making predictions
  Ensure the first sequence in your MSA does not contain gaps.
  If it does, remove the corresponding columns.
