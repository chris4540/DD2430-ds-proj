# Data Science Project
Group 15 - Fashion Image similarity

## Target for week 46
- [ ] Implement and test with fashion minst
  - [x] Baseline implmentation
  - [x] Investigate ignite
  - [x] Use ignite
  - [ ] Implment Siamese Embedding (Siam),  Siam+Cat, Siam+Cat Cos
- [x] Write script to download deep fashion dataset
- [ ] Have dataset and pairing for deep fashion dataset
- [ ] Implment on Deep fashion
- [ ] Deploy and babysit a while (Target by Sat)

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

#### Useful tool
1. https://github.com/fastai/fastai
2. https://github.com/williamFalcon/pytorch-lightning

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
2. TSNE from *cudf* library runs on GPU and usage same as scikit.
   https://github.com/rapidsai/cudf
   https://towardsdatascience.com/600x-t-sne-speedup-with-rapids-5b4cf1f62059
   (Has installation issues due to cuda requirement. Can be done in seperate VM in worst case. (Not current Priority))
   https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
   https://github.com/CannyLab/tsne-cuda

##### Getting Similar Images
1. Using indexing from *ANNOY* library results is fastest nearest neighbour retreival.
  https://jovian.ml/gautham20/deepfashion-similar-images-annoy
  (easy to use (Not a Priority))
