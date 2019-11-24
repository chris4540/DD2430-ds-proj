# Data Science Project
Group 15 - Fashion Image similarity

# How to run tests
-------------------
```bash
$ pip install pytest pytest-timeout
```
```bash
$ pytest tests
```
# Preparation
------------------
1. Download the image data
```bash
$ scripts/download_deepfashion_ds.sh
```
2. Build the metadata csv
```bash
$ scripts/create_deepfashion_meta.py
```
## Target for week 46
- [x] Investigate the torch training framework
- [x] Implement and test with fashion minst
  - [x] Baseline implmentation
  - [x] Investigate ignite
  - [x] Use ignite
- [x] Write script to download deep fashion dataset
- [x] Install and test cuML for t-SNE visualization

## Target for week 47
- [x] Preprocess the deep fashion data. Did the mean and std calculation
- [x] Implment Siamese Embedding (Siam)
- [ ] Implment Siam+Cat, Siam+Cat Cos
- [ ] Implment Siam+Cat Cos
- [x] Have dataset and pairing for deep fashion dataset. TODO: The validation and test
- [x] Implment on Deep fashion
- [ ] Deploy and babysit a while
- [ ] Automatic backup experiment directory
- [ ] Create detailed attributes for a class. Person in charge: Hassan
- [ ] KNN searching (ANNOY) and top-k evaluation. Out-training evalution. Person in charge: Hassan
- [x] Model candidate: resnet 18
- [ ] Model candidate: vgg16 (wo bn)
- [ ] Search classical unsupervised learning method for meansuring cluster distances
- [ ] Balance sampling from the train dataset. Scripts to generate the metadata.csv
- [x] Each epoch supervised evaluation
- [ ] Each epoch unspervised evaluation


## Reason to use ignite
1. Supported by pytorch offical
2. Similar to keras
3. Stable releases
4. Flexible callback system



## Available Implementation and literature
--------------------------------------------
## [Orginal Paper](https://www.cs.cornell.edu/~kb/publications/SIG15ProductNet.pdf)

#### Similar Siamese work:

+ https://blogs.technet.microsoft.com/machinelearning/2018/07/10/how-to-use-siamese-network-and-pre-trained-cnns-for-fashion-similarity-matching/
- https://github.com/erikamenezes/CS231nFinalProject/tree/master/Experiment3_SiameseNet

+ https://towardsdatascience.com/siamese-networks-and-stuart-weitzman-boots-c414be7eff78
- https://github.com/sugi-chan/shoes_siamese/blob/master/Siamese-clothes.ipynb

+ https://www.linkedin.com/pulse/using-deep-learning-fashion-similarity-face-vaibhav-gusain

#### Resnet Classificaton style:

+ https://towardsdatascience.com/similar-images-recommendations-using-fastai-and-annoy-16d6ceb3b809
- https://jovian.ml/gautham20/deepfashion-similar-images-annoy
+ https://blog.floydhub.com/similar-fashion-images/
- https://github.com/khanhnamle1994/fashion-recommendation
+ https://github.com/PlabanM1/FashionNet/blob/master/FashionNet.ipynb

#### Implmentaion Reference
- Keras
  - https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
  - https://github.com/sugi-chan/shoes_siamese/blob/master/Siamese-clothes.ipynb
  - https://github.com/PlabanM1/FashionNet/blob/master/FashionNet.ipynb

- PyTorch
  - https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
  - https://github.com/fangpin/siamese-pytorch/blob/5543f1e844964b116dc9d347a5eb164c6a7afe6d/model.py#L6
  - https://github.com/adambielski/siamese-triplet/blob/master/Experiments_FashionMNIST.ipynb
  - Refer to my notebooks for required changes to siamese to allign them to the paper

##### Baseline
1. https://jovian.ml/gautham20/deepfashion-similar-images-annoy
    Just using classification
2. Type b, c, d Siamese network in the paper.

##### Visualization
1. Scikit T-SNE option very slow.
   https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
2. TSNE from *cuML* library runs on GPU and usage same as scikit.
   https://rapids.ai/start.html#rapids-release-selector
   https://towardsdatascience.com/600x-t-sne-speedup-with-rapids-5b4cf1f62059
3. TSNE-CUDA
   https://github.com/CannyLab/tsne-cuda
   https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
*  Notes: Check cuda version
  ```bash
  cat /usr/local/cuda/version.txt
  nvcc --version
  ```
4. Parallel version of t-SNE
  https://github.com/DmitryUlyanov/Multicore-TSNE
  ```bash
  conda install -c powerai multicoretsne
  ```

##### Getting Similar Images
1. Using indexing from *ANNOY* library results is fastest nearest neighbour retreival.
  https://jovian.ml/gautham20/deepfashion-similar-images-annoy
  (easy to use (Not a Priority))

##### Customized Google Clould image
1. pytorch-1-3-rapids-0-10-cu100-20181117
