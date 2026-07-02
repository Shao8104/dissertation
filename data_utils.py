from torch.utils.data import Subset


def create_label_noniid_subset(dataset, client_id):
    """
    Create label-based Non-IID data split for MNIST.

    Client 0: digits 0, 1
    Client 1: digits 2, 3
    Client 2: digits 4, 5
    Client 3: digits 6, 7
    Client 4: digits 8, 9
    """

    label_groups = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
    }

    if client_id not in label_groups:
        raise ValueError(
            f"Invalid client_id: {client_id}. "
            "Expected client_id in [0, 1, 2, 3, 4]."
        )

    allowed_labels = label_groups[client_id]

    targets = dataset.targets

    indices = [
        idx
        for idx, label in enumerate(targets)
        if int(label) in allowed_labels
    ]

    return Subset(dataset, indices), allowed_labels