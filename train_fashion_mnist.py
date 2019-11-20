from trainer.siamese import SiameseFashionMNISTTrainer
from sklearn.manifold import TSNE
from utils.plot_fashion_minst import plot_embeddings

# hyper_params = dict()
trainer = SiameseFashionMNISTTrainer(
    log_interval=5, lr=1e-2, epochs=2, batch_size=100)
trainer.run()

embeddings, labels = trainer.map_val_ds_to_emb_space()

tsne = TSNE(random_state=1, n_iter=1000, metric="euclidean")

projected_emb = tsne.fit_transform(embeddings)
fig = plot_embeddings(projected_emb, labels)

fig.savefig('fashion_mnist.png', bbox_inches='tight')
