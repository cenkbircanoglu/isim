import os
from io import BytesIO

import scipy.misc
import tensorflow as tf

from utils import log_loss_summary

try:
    FileWriter = tf.compat.v1.summary.FileWriter
    Summary = tf.compat.v1.Summary
except:
    FileWriter = tf.summary.FileWriter
    Summary = tf.Summary

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def log_losses(
    logger, step, phase, cls_losses_list, ap_scores, total_cnt, seg_losses_list=None
):
    log_loss_summary(logger, cls_losses_list[0], step, tag=f"{phase}_total_cls_loss")
    for idx, tr_loss in enumerate(cls_losses_list):
        if idx == 0:
            continue
        if sum(cls_losses_list[idx]) != 0:
            log_loss_summary(
                logger, cls_losses_list[idx], step, tag=f"{phase}_block{idx}_cls_loss"
            )

    if seg_losses_list is not None:
        log_loss_summary(
            logger, seg_losses_list[0], step, tag=f"{phase}_total_seg_loss"
        )
        for idx, tr_loss in enumerate(seg_losses_list):
            if idx == 0:
                continue
            if sum(seg_losses_list[idx]) != 0:
                log_loss_summary(
                    logger,
                    seg_losses_list[idx],
                    step,
                    tag=f"{phase}_block{idx}_seg_loss",
                )

    log_loss_summary(logger, float(ap_scores[0]) / total_cnt, step, tag=f"{phase}_mAP")
    for idx, ap_score in enumerate(ap_scores):
        if idx == 0:
            continue
        if ap_scores[idx] != 0:
            log_loss_summary(
                logger,
                float(ap_scores[idx]) / total_cnt,
                step,
                tag=f"{phase}_block{idx}_mAP",
            )


class Logger(object):
    def __init__(self, log_dir):
        self.writer = FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        scipy.misc.toimage(image).save(s, format="png")

        # Create an Image object
        img_sum = Summary.Image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        summary = Summary(value=[Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )

            # Create a Summary value
            img_summaries.append(
                Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
            )
        # Create and write Summary
        summary = Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
