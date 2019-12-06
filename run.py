from experiment.siamcos import SiameseCosDistance

exp = SiameseCosDistance(
    # debug=True,
    lr=5e-4, weight_decay=1e-5, eta_min=1e-6, batch_size=128)

exp.run(max_epochs=5)
