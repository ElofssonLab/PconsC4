# PconsC4:
Fast, accurate, and hassle-free contact prediction.

## Installation instructions:

Download the tarball from the [releases tab](https://github.com/ElofssonLab/PconsC4/releases)

    pip3 install numpy Cython pythran &&
    pip3 install pconsc4-0.2.tar.gz

NB: the trained model is a bit over Github's limit, so they cannot be checked in the repo, hence the need for the tarball.

You will also need a deep learning backend compatible with Keras. We recommend Tensorflow:

    pip3 install -U tensorflow



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

We also provide a function to format the output in CASP format: 

    # Save in CASP format:
    from pconsc4.utils import format_contacts_casp
    print(format_contacts_casp(pred_2['cmap'], seq_2, min_sep=5))

## Troubleshooting:

If pyGaussDCA fails to install with template errors, upgrade your compiler. GCC 5 and higher is known to work.
