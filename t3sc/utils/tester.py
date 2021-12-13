from collections import defaultdict
import logging
import os
from PIL import Image

import json
import matplotlib
import numpy as np
import torch
import time

from t3sc.models.metrics import (
    mergas,
    mfsim,
    mpsnr,
    msam,
    mssim,
    psnr,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RGB_DIR = "rgb"
RGB_CROP_DIR = "rgb_cropped"


def log_metrics(metrics, log_in=True):
    inout_metrics = list(
        set(
            [
                n.split("_")[0]
                for n in metrics.keys()
                if n.split("_")[1] in ["in", "out"]
            ]
        )
    )
    for name, value in metrics.items():
        if name.split("_")[0] in inout_metrics:
            continue
        if isinstance(value, list):
            value = value[-1]
        logger.info(f"\t{name.upper()} : {value:.4f}")

    for m_name in inout_metrics:
        m_out = metrics[f"{m_name}_out"]
        if log_in:
            m_in = metrics[f"{m_name}_in"]
        else:
            m_in = 0
        if isinstance(m_out, list):
            if log_in:
                m_in = m_in[-1]
            m_out = m_out[-1]
        logger.info(f"\t{m_name.upper()} : in={m_in:.4f}, out={m_out:.4f}")


class Tester:
    def __init__(
        self,
        save_rgb,
        save_rgb_crop,
        save_raw,
        save_labels,
        seed,
        idx_test,
    ):
        self.save_rgb = save_rgb
        self.save_rgb_crop = save_rgb_crop
        self.save_raw = save_raw
        self.save_labels = save_labels
        self.seed = seed
        self.idx_test = idx_test

    def eval(self, model, datamodule):
        torch.manual_seed(self.seed)

        self.metrics = {"n_params": model.count_params()[0]}
        self.all_metrics = defaultdict(list)
        if model.__class__.__name__ == "SPCNN_TF":
            dev = "cpu"
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(dev)
        model = model.to(device)

        self.extract_info(datamodule)

        logger.debug(f"Iterating on batches")

        # evaluate on the whole test set
        datamodule.max_test = None
        datamodule.idx_test = self.idx_test
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()
        n_batches = len(test_dataloader)
        for i, batch in enumerate(test_dataloader):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logger.info(f'Image ID {i+1}/{n_batches}: {batch["img_id"][0]}')

            model.eval()
            x = batch["x"]
            logger.debug(f"x input shape : {x.shape}")
            with torch.no_grad():
                tic = time.time()

                if model.ssl:
                    logger.info(f"ssl test")
                    bs, c, h, w = batch["x"].shape
                    out = torch.zeros_like(batch["x"])
                    N = int(np.ceil(c / model.n_ssl))
                    for i in range(N):
                        ssl_idx = model.get_ssl_idx(i, c).long()
                        batch["ssl_idx"] = ssl_idx
                        if (
                            model.block_inference
                            and model.block_inference.use_bi
                        ):
                            _out = model.forward_blocks(**batch)
                        else:
                            _out = model.forward(**batch)
                        out[:, ssl_idx] = _out[:, ssl_idx]
                    out = out.float()
                else:
                    if model.block_inference and model.block_inference.use_bi:
                        out = model.forward_blocks(**batch)
                    else:
                        out = model.forward(**batch)
                elapsed = time.time() - tic
                batch["out"] = out.clamp(0, 1)

            logger.debug(f"Inference done")
            self.all_metrics["inference_time"].append(elapsed)

            self.compute_metrics_denoising(**batch)
            logger.info(f"Image metrics :")

            img_id = batch["img_id"][0]

            crop_info = self.get_crop_info(img_id)
            if len(crop_info) == 0:
                logger.debug(f"No crop found for {img_id}, not saving to RGB")
                logger.debug(f"{self.img_info}")
                continue

            if self.save_rgb:
                self._save_rgb(**batch)
        self.aggregate_metrics()

    def get_crop_info(self, img_id):
        try:
            return self.img_info[img_id]["crop"]
        except KeyError:
            return []

    def extract_info(self, datamodule):
        logger.debug("Extracting datamodule info..")
        crops = datamodule.dataset_factory.CROPS
        rgb = datamodule.dataset_factory.RGB
        self.img_info = {
            img_id.replace(".", ""): {"crop": crop, "rgb": rgb}
            for (img_id, crop) in crops.items()
        }

    def to_pil(self, x, img_id, crop=False):
        bands = self.img_info[img_id]["rgb"]
        bands = torch.tensor(bands).long()
        x = x[0, bands].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(np.uint8(255 * x.clip(0, 1)))
        if crop:
            img = img.crop(self.img_info[img_id]["crop"])
        return img

    def compute_metrics_denoising(self, x, y, out, img_id, **kwargs):
        # x : (bs, c, h, w)
        logger.debug("Computing denoising metrics..")
        x = x.clamp(0, 1)
        if torch.any(torch.isnan(y)):
            logger.debug(f"Nan detected in y")
        if torch.any(torch.isnan(x)):
            logger.debug(f"Nan detected in x")
        img_metrics = {}
        with torch.no_grad():
            logger.debug("Computing PSNR")
            img_metrics["psnr_in"] = psnr(x, y).item()
            img_metrics["psnr_out"] = psnr(out, y).item()

            logger.debug("Computing MPSNR")
            img_metrics["mpsnr_in"] = mpsnr(x, y).item()
            img_metrics["mpsnr_out"] = mpsnr(out, y).item()

            logger.debug("Computing MSSIM")
            img_metrics["mssim_in"] = mssim(x, y).item()
            img_metrics["mssim_out"] = mssim(out, y).item()

            h, w = x.shape[-2:]
            s = min(h, w)
            logger.debug(f"Computing MFSIM (s={s})")

            img_metrics["mfsim_in"] = mfsim(
                x[:, :, :s, :s].float(), y[:, :, :s, :s].float()
            ).item()
            img_metrics["mfsim_out"] = mfsim(
                out[:, :, :s, :s].float(), y[:, :, :s, :s].float()
            ).item()

            logger.debug("Computing MERGAS")
            img_metrics["mergas_in"] = mergas(x, y).item()
            img_metrics["mergas_out"] = mergas(out, y).item()

            logger.debug("Computing MSAM")
            img_metrics["msam_in"] = msam(x, y).item()
            img_metrics["msam_out"] = msam(out, y).item()

        log_metrics(img_metrics)
        for k, v in img_metrics.items():
            self.all_metrics[k].append(v)

        self.metrics[img_id[0]] = img_metrics

    def aggregate_metrics(self):
        global_metrics = {}
        for name, samples in self.all_metrics.items():
            global_metrics[name] = np.mean(samples)
        self.metrics["global"] = global_metrics

        logger.info("-" * 16)
        logger.info("Global metrics :")
        log_metrics(global_metrics)

        with open("test_metrics.json", "w") as f:
            f.write(json.dumps(self.metrics))
        logger.info("Metrics saved to 'test_metrics.json'")
        logger.info(f"Current workdir : {os.getcwd()}")

    def _save_rgb(self, x, out, img_id, y=None, crop=False, **kwargs):
        logger.debug(f"Trying to save RGB")
        img_id = img_id[0]
        folder = RGB_CROP_DIR if crop else RGB_DIR
        os.makedirs(folder, exist_ok=True)
        img_pil = {
            "in": self.to_pil(x, img_id, crop=crop),
            "out": self.to_pil(out, img_id, crop=crop),
        }

        for (cat, pil) in img_pil.items():
            path_img = f"{folder}/{img_id}_{cat}.png"
            pil.save(path_img)
            logger.debug(f"Image saved to {path_img!r}")
