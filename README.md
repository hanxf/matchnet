# Matchnet

Matchnet is a deep learning approach for patch-based local image matching, which
jointly learns feature representation and matching function from data. More
details about this approach can be found in our
[CVPR'15 paper](http://www.cs.unc.edu/~xufeng/cs/papers/cvpr15-matchnet.pdf).

This repository contains reference source code for evaluating MatchNet models on
[Phototour Patch dataset](http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html).

Below is a step-by-step guide for downloading the dataset, generate patch
database and running evaluation with Matchnet models. We assume
[Caffe](http://caffe.berkeleyvision.org) is installed (preferably with GPU
support) and Pycaffe (Caffe's python interface) is also installed and added to
PYTHONPATH.

Clone the repository.

    git clone https://github.com/hanxf/matchnet.git
    cd matchnet

Downlowd the Phototour patch dataset. 

    cd data/phototour

    curl -O http://www.cs.ubc.ca/~mbrown/patchdata/liberty.zip
    unzip -q -d liberty liberty.zip
    rm liberty.zip

    curl -O http://www.cs.ubc.ca/~mbrown/patchdata/notredame.zip
    unzip -q -d notredame notredame.zip
    rm notredame.zip

    curl -O http://www.cs.ubc.ca/~mbrown/patchdata/yosemite.zip
    unzip -q -d yosemite yosemite.zip
    rm yosemite.zip

    cd ../..

Generate leveldb database for each dataset. The databases are saved under `data/leveldb`.

    ./run_gen_data.sh

Download pretrained Matchnet models. (Here we only download the model trained on liberty. For more models see `models/README.md`

    cd models

    curl -O http://cs.unc.edu/~xufeng/matchnet/models/liberty_r_0.01_m_0.feature_net.pb
    curl -O http://cs.unc.edu/~xufeng/matchnet/models/liberty_r_0.01_m_0.classifier_net.pb

    cd ..

Evalute the liberty model on notredame's test set. (Remove the quoted argument to use CPU.)

    ./run_eval.sh liberty notredame "--use_gpu --gpu_id=0"

When the script is done, the last line should be the following:

    Error rate at 95% recall: 4.48%
    
## License and Citation

Matchnet source code is released under BSD license. The reference models are released for unrestriced use.

Please cite our paper if Matchnet helps your research:

    @inproceedings{matchnet_cvpr_15,
      Author = {Han, Xufeng and Leung, Thomas and Jia, Yangqing and Sukthankar, Rahul and Berg, Alexander. C.},
      Booktitle = {CVPR},
      Title = {MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching},
      Year = {2015}
    } 
