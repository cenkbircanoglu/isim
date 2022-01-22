import logging
from functools import partial
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loaders import my_collate
from misc.generate_label import generate_pseudo_label


# def _work(process_id, model, dataset_split, dataset, cfg):
#     n_gpus = torch.cuda.device_count()
#     databin = dataset_split[process_id]
#
#     data_loader = DataLoader(
#         databin,
#         shuffle=False,
#         num_workers=cfg.workers // n_gpus,
#         pin_memory=False,
#         persistent_workers=True,
#     )
#     results = []
#     with torch.no_grad(), cuda.device(process_id):
#         print("\nCreating Pseudo Labels for training")
#         model.eval()
#         model.cuda()
#         for item in tqdm(data_loader, total=len(data_loader.dataset)):
#             idx = item["idx"][0]
#             img_i = [img_ii.cuda(non_blocking=True) for img_ii in item["img"]]
#             res = generate_pseudo_label(
#                 model,
#                 img_i,
#                 item["label"][0],
#                 item["size"],
#                 fg_thres=cfg.cam_eval_thres,
#             )
#             results.append((idx.cpu().item(), res))
#     with Pool(processes=int((cpu_count() // 3) * 2)) as pool:
#         print("Applying CRF to CAM results")
#         pool.starmap(dataset.update_cam, tqdm(results, total=len(results)))


def updater(queue, dataset=None, fg_thres=None):
    while True:
        msg = queue.get()
        if msg == "DONE":
            break
        idx, cam = msg
        dataset.update_cam(idx, cam, fg_thres=fg_thres)


def calculate_crf(
        model,
        cfg,
        tr_loader,
        val_loader,
        dataset_train,
        dataset_valid,
        device,
        crf_batch_size=500,
        *args,
        **kwargs
):
    model.eval()
    tr_results = []
    # val_results = []

    with torch.set_grad_enabled(False):
        logging.info("\nCreating Pseudo Labels for training")
        for item in tqdm(tr_loader, total=len(tr_loader.dataset)):
            idx = item["idx"][0]
            img_i = [img_ii.to(device) for img_ii in item["img"]]
            res = generate_pseudo_label(
                model, img_i, item["label"][0], item["size"]
            )
            tr_results.append((idx.cpu().item(), res))
            if len(tr_results) >= crf_batch_size:
                with Pool(processes=4) as pool:
                    pool.starmap(
                        partial(
                            dataset_train.update_cam, fg_thres=cfg.cam_eval_thres
                        ),
                        tqdm(tr_results, total=len(tr_results)),
                    )
                tr_results = []
        # print("\nCreating Pseudo Labels for validation")
        # for item in tqdm(val_loader, total=len(val_loader.dataset)):
        #     idx = item["idx"][0]
        #     img_i = [img_ii.to(device) for img_ii in item["img"]]
        #     res = generate_pseudo_label(model, img_i, item["label"][0], item["size"])
        #     val_results.append((idx.cpu().item(), res))
        #     if len(val_results) >= crf_batch_size:
        #         with Pool(processes=4) as pool:
        #             pool.starmap(
        #                 partial(dataset_valid.update_cam, fg_thres=cfg.cam_eval_thres),
        #                 tqdm(val_results, total=len(val_results)),
        #             )
        #         val_results = []
    logging.info("Applying CRF to CAM results")
    if len(tr_results) > 0:
        with Pool(processes=8) as pool:
            pool.starmap(
                partial(dataset_train.update_cam, fg_thres=cfg.cam_eval_thres),
                tqdm(tr_results, total=len(tr_results)),
            )
    # if len(val_results) > 0:
    #     with Pool(processes=4) as pool:
    #         pool.starmap(
    #             partial(dataset_valid.update_cam, fg_thres=cfg.cam_eval_thres),
    #             tqdm(val_results, total=len(val_results)),
    #         )

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    return {"train": loader_train, "valid": loader_valid}
