from .siamcos import SiameseCosDistanceCat


class SiameseEucDistanceCat(SiameseCosDistanceCat):
    l2_normalize = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def scale_factor(self):
        """
        The scale factor used to scale the sum of cross entorpy term to
        make contractive loss and cross entorpy term comparable.

        Consider:
            loss = contras_loss + clsf_loss1 + clsf_loss2
        is not ideal as contras_loss varies as the margin varies.

        A better case would be:
            loss = contras_loss + (clsf_loss1 + clsf_loss2) * scale_factor

        We consider an extreme case that min(contras_loss) ~ (m**2) * (0.5).
        That is all distances among the embedding vectors are zeros.
        """
        ret = 0.5 * (self.margin ** 2)
        return ret
