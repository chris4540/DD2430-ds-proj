from trainer.siamese import SiameseTrainer
from cuml.manifold import TSNE
import pickle

trainer = SiameseTrainer(log_interval=1, lr=5e-2, epochs=20, batch_size=100)
trainer.run()
trainer.save_model()
embeddings, labels = trainer.map_train_ds_to_emb_space()


tsne = TSNE(n_iter=1000, metric="euclidean")
projected_emb = tsne.fit_transform(embeddings)


with open('projected_emb.pkl', 'wb') as handle:
    pickle.dump(projected_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels.pkl', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fig = plot_embeddings(projected_emb, labels)
# fig.savefig('fashion_mnist.png', bbox_inches='tight')
