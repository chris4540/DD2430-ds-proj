# from trainer.siamese import SiameseTrainer
# from cuml.manifold import TSNE
# import pickle
# import sys

# trainer = SiameseTrainer(
#     exp_folder="./exp_folders/exp_siamese",
#     log_interval=1, lr=5e-2, epochs=10, batch_size=256)
# trainer.run()
# sys.exit(0)
# trainer.save_model()
# embeddings, labels = trainer.map_train_ds_to_emb_space()


# tsne = TSNE(n_iter=1000, metric="euclidean")
# projected_emb = tsne.fit_transform(embeddings)


# with open('projected_emb.pkl', 'wb') as handle:
#     pickle.dump(projected_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('labels.pkl', 'wb') as handle:
#     pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # fig = plot_embeddings(projected_emb, labels)
# # fig.savefig('fashion_mnist.png', bbox_inches='tight')
from experiment.siamese import Siamese
import numpy as np
for m_sq in [100, 1, 10]:
    margin = np.round(np.sqrt(m_sq), decimals=4)
    exp = Siamese(
        "./exp_folders/exp_siamese_m2_{}".format(m_sq),
        lr=0.0005,
        weight_decay=1e-5,
        eta_min=1e-6,
        epochs=20,
        batch_size=128,
        margin=margin)
    exp.run()
