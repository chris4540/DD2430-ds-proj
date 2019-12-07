from .siamcos import SiameseCosDistanceWithCat


class SiameseEucDistanceWithCat(SiameseCosDistanceWithCat):
    l2_normalize = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale_factor = kwargs.get("scale_factor", None)

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

        # if we fix the scale_factor
        if self._scale_factor is not None:
            return self._scale_factor

        lambda_ = self.hparams.lambda_
        ret = lambda_ * 2 / (self.margin ** 2)
        return ret
