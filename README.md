<div align="center">
<h2>PrimeDepth: Efficient Monocular Depth Estimation with a Stable Diffusion Preimage</h2>

[**Denis Zavadski**](https://scholar.google.com/citations?user=S7mDg00AAAAJ)<sup>\*</sup>&emsp;·&emsp;[**Damjan Kalšan**](https://scholar.google.com/citations?user=6NAxnFUAAAAJ)<sup>\*</sup>&emsp;·&emsp;[**Carsten Rother**](https://scholar.google.com/citations?user=N_YNMIMAAAAJ)

Computer Vision and Learning Lab,<br/>
IWR, Heidelberg University

<sup>*</sup>equal contribution

<strong>ACCV 2024</strong>

<a href='https://vislearn.github.io/PrimeDepth/'><img src='https://img.shields.io/badge/Project_Page-PrimeDepth-green' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2409.09144"><img src='https://img.shields.io/badge/arXiv-PDF-red' alt='Paper PDF'></a> <a href="https://github.com/vislearn/PrimeDepth"><img src='https://img.shields.io/badge/Github-Code-blue' alt='Github Code'></a>

PrimeDepth is a diffusion-based monocular depth estimator which leverages the rich representation of the visual world stored within Stable Diffusion. The representation, termed <q>preimage</q>, is extracted in a single diffusion step from frozen Stable Diffusion 2.1 and adjusted towards depth prediction. PrimeDepth yields detailed predictions while simulatenously being fast at inference time due to the single-step approach.

</div>

![teaser](images/teaser.png)

## Introduction
This is an inference codebase for [PrimeDepth](https://arxiv.org/abs/2409.09144) based on <a href="https://github.com/Stability-AI/stablediffusion">Stable Diffusion 2.1</a>. Further details and visual examples can be found on the [project page](https://vislearn.github.io/PrimeDepth/).

## Installation

1. Create and activate a virtual environment:
    ```
    conda create -n PrimeDepth python=3.9
    conda activate PrimeDepth
    ```

2. Install dependencies:
    ```
    pip3 install -r requirements.txt
    ```

3. Download the [weights](https://huggingface.co/CVL-Heidelberg/PrimeDepth)

4. Adjust the attribute `ckpt_path` in `configs/inference.yaml` to point to the downloaded weights from the previous step

## Usage

```
from scripts.utils import InferenceEngine


config_path = "./configs/inference.yaml"
image_path = "assets/bober.jpg"

ie = InferenceEngine(pd_config_path=config_path, device="cuda")

depth_ssi, depth_color = ie.predict(image_path)
```

PrimeDepth predicts in inverse space. The raw model predictions are stored in `depth_ssi`, while a colorized prediction `depth_color` is precomputed for visualization convenience:

```
depth_color.save("demo.png")
```

## Citation
```bibtex
@misc{zavadski2024primedepth,
    title={PrimeDepth: Efficient Monocular Depth Estimation with a Stable Diffusion Preimage}, 
    author={Denis Zavadski and Damjan Kalšan and Carsten Rother},
    year={2024},
    eprint={2409.09144},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.09144}, 
}
```