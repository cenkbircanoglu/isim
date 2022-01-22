import hydra
from torch.utils.data import DataLoader

from dataset_loaders import seg_loader, my_collate
from misc.cal_crf import async_calculate_crf
from models import initialize_model


@hydra.main(config_path="./conf/", config_name="train")
def main(cfg):
    (
        dataset_train,
        dataset_valid,
        tr_data_scaled,
        val_data_scaled,
    ) = seg_loader.data_loaders(cfg)

    model = initialize_model(cfg)
    tr_loader = DataLoader(
        tr_data_scaled,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=my_collate,
    )
    val_loader = DataLoader(
        val_data_scaled,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=my_collate,
    )
    loaders = async_calculate_crf(
        model,
        cfg,
        tr_loader,
        val_loader,
        dataset_train,
        dataset_valid,
        "cuda",
        cam_eval_thres=cfg.cam_eval_thres,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )


if __name__ == "__main__":
    main()
