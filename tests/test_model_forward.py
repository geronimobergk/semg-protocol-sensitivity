import torch

from tinyml_semg_classifier.models.st_cnn_gn import ST_CNN_GN


def test_st_cnn_forward_shape():
    model = ST_CNN_GN(
        num_electrodes=2,
        num_samples=8,
        conv_channels=[4],
        kernel_size=(3, 3),
        pool_sizes=[(1, 1)],
        conv_dropout=0.0,
        gn_groups=1,
        head_hidden=[8, 4],
        head_dropout=0.0,
        num_classes=2,
    )
    xb = torch.zeros((3, 2, 8), dtype=torch.float32)
    logits = model(xb)
    assert logits.shape == (3, 2)
