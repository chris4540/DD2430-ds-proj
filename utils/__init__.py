import torch
import numpy as np

USING_CUDA = torch.cuda.is_available()


def map_images_to_embbedings(model, images):
    """
    With the deep network, project images / batch of images to embedding space
    """
    assert len(images.shape) == 4

    # turn the model to evaluation mode
    model.eval()
    with torch.no_grad():
        if USING_CUDA:
            images = images.cuda()

        ret = model(images)

    ret = ret.cpu()

    return ret


def extract_embeddings(model, dataloader):
    """
    TODO:
    model should have attr emb_net and use it
    """

    emb_list = []
    label_list = []
    for imgs, lbls in dataloader:
        emb_list.append(map_images_to_embbedings(model, imgs).numpy())
        label_list.append(lbls.numpy())

    embeddings = np.concatenate(emb_list, axis=0)
    labels = np.concatenate(label_list)

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
