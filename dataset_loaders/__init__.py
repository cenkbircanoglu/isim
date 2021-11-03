import torch.utils.data


def my_collate(batch):
    new_batch = []
    for item in batch:
        item.pop("original_label", None)
        new_batch.append(item)
    return torch.utils.data.dataloader.default_collate(new_batch)
