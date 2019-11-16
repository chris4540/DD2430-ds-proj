class Custom_ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embd_output1, class_output1, embd_output2, class_output2, label, class1, class2):
        euclidean_distance = F.pairwise_distance(
            embd_output1, embd_output1, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        loss_1 = nn.CrossEntropyLoss(class_output1, class1)
        loss_2 = nn.CrossEntropyLoss(class_output2, class2)

        return loss_contrastive+loss_1+loss_2
