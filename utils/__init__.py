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

    emb_list = []
    label_list = []
    for imgs, lbls in dataloader:
        emb_list.append(map_images_to_embbedings(model, imgs).numpy())
        label_list.append(lbls.numpy())

    embeddings = np.concatenate(emb_list, axis=0)
    labels = np.concatenate(label_list)

    return embeddings, labels
