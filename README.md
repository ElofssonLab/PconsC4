# PconsC4:
fast, accurate, and hassle-free contact prediction.

##Installation instructions:

    pip3 install numpy Cython &&
    pip3 install -U git+https://github.com/serge-sans-paille/pythran@master &&
    pip3 install -U git+https://github.com/ElofssonLab/pyGaussDCA.git@master &&
    pip3 install .

You will also need a deep learning backend compatible with Keras. We recommend Tensorflow:

    pip3 install -U tensorflow


## Usage instructions

Inside Python:

    import pconsc4

    model = pconsc4.get_pconsc4()

    pred_1 = pconsc4.predict(model, 'path/to/alignment1')
    pred_2 = pconsc4.predict(model, 'path/to/alignment2')

