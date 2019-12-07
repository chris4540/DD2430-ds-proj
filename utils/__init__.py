"""
Implementation using pytorch ignite

Reference:
    https://github.com/pytorch/ignite/blob/v0.2.1/examples/mnist/mnist.py
    https://fam-taro.hatenablog.com/entry/2018/12/25/021346

TODO:
    resume from checkpoint (check statedict)
"""
import torch
import numpy as np
from tqdm import tqdm


def map_imgs_to_embs(model, images, device):
    """
    With the deep network, project images / batch of images to embedding space
    """
    assert len(images.shape) == 4

    # turn the model to evaluation mode
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        ret = model(images)
    return ret


def extract_embeddings(emb_net, dataloader, device="cuda"):
    """
    TODO:
    model should have attr emb_net and use it
    """

    emb_net.to(device)

    emb_list = []
    label_list = []
    for imgs, lbls in tqdm(dataloader, desc='Extract emb vecs'):
        # extract embedding vector
        emb_vecs = map_imgs_to_embs(emb_net, imgs, device)
        # save the value to a list
        emb_list.append(emb_vecs)
        label_list.append(lbls)

    embeddings = torch.cat(emb_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    return embeddings, labels


class PairIndexSet:

    def __init__(self):
        self._set = set()

    @staticmethod
    def _as_ordered_tuple(tuple_):
        """
        Make a tuple to have smaller element at the first place

        E.g.:
        _as_ordered_tuple((2, 1))
        # (1, 2)
        _as_ordered_tuple((1, 100))
        # (1, 100)
        """
        a, b = tuple_
        if a > b:
            a, b = b, a  # swap
        return a, b

    def __contains__(self, elem):
        if not isinstance(elem, tuple):
            raise ValueError("Input must be tuple")
        # reorder if a > b
        a, b = self._as_ordered_tuple(elem)
        # check if in the internal set
        ret = ((a, b) in self._set)
        return ret

    def contains(self, elem):
        return self.__contains__(elem)

    def __repr__(self):
        return repr(self._set)

    def add(self, elem):
        """
        Add element elem to the set.
        """
        self._set.add(self._as_ordered_tuple(elem))

    def clear(self):
        """
        Remove all elements from the set.
        """
        self._set.clear()

    def __len__(self):
        return len(self._set)
