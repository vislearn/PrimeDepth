import os
import torch
import safetensors.torch
import matplotlib
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as tt
from omegaconf import OmegaConf
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(
                ckpt_path, map_location=torch.device(location)
            )
        )
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path, device="cpu"):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).to(device)
    print(f'Loaded model config from [{config_path}]')
    return model


def get_image(path):
    image = Image.open(path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image


class InferenceEngine:
    """
    Utility class for obtaining PrimeDepth predictions

    """

    def __init__(self, pd_config_path, blip2_cache_dir=None, cmap="Spectral", device="cuda"):
        """
        pd_config_path : str
            Path to the model configuration file
        blip2_cache_dir : str, optional
            Path to the cache directory for the BLIP2 model, by default None
        cmap: str, optional
            Matplotlib colormap name, by default "Spectral"
        device : str, optional
            Device to run the model on, by default "cuda"

        """

        self.pd = create_model(pd_config_path, device)
        self.processor, self.model = self.load_BLIP2(blip2_cache_dir, device)
        self.cm = matplotlib.colormaps[cmap]

    def load_BLIP2(self, cache_dir=None, device="cuda"):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir=cache_dir
            )
        model = model.to(device)
        return processor, model

    def captionize(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

    def predict(self, image_path, max_size=1024):
        """
        Parameters
        ----------
        image_path : str
            Path to the image
        max_size : int, optional
            Maximal processing size of the longer image edge, by default 1024

        Returns
        -------
        depth_ssi : np.ndarray
            Scale and shift invariant depth map prediction
        depth_color : PIL.Image
            Colorized depth map prediction

        """

        image = get_image(image_path)
        caption = self.captionize(image)
        w, h = image.size

        if max_size is not None and max(h, w) > max_size:
            if h == w:
                image = tt.Resize(max_size)(image)
            else:
                image = tt.Resize(max_size-1, max_size=max_size)(image)

        with torch.no_grad():
            labels = self.pd.get_label_from_image(image, prompt=caption)
        depth_ssi = labels['depth'].clone().mean(dim=1)

        if max_size is not None and max(h, w) > max_size:
            depth_ssi = F.interpolate(depth_ssi[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        else:
            depth_ssi = depth_ssi[0]

        depth_ssi = depth_ssi.cpu().numpy().astype(np.float32)

        depth_color = depth_ssi.copy()
        depth_color = (depth_color - depth_color.min()) / (depth_color.max() - depth_color.min())
        depth_color = self.cm(depth_color)[:, :, :3]
        depth_color = (depth_color * 255).astype(np.uint8)
        depth_color = Image.fromarray(depth_color)

        return depth_ssi, depth_color
