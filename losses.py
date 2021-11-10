import numpy as np
import torch
import torch.nn as nn

mlsm_loss = nn.MultiLabelSoftMarginLoss()
mce_loss = nn.CrossEntropyLoss()


def calculate_label_weights(labels):
    weights = torch.sum(labels, axis=[1, 2])
    weights[weights != 0] = 1
    weights = weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return weights


def modified_cross_entropy_loss(seg_preds, labels):
    weights = calculate_label_weights(labels)
    labels = labels.long()

    return mce_loss(seg_preds * weights, labels)


if __name__ == "__main__":
    label = torch.from_numpy(
        np.array(
            [
                [
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [1, 2, 0, 0]],
                ],
                [
                    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                    [[1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
                ],
            ]
        )
    )
    print(label.shape)
    print(calculate_label_weights(label))
