Textual Perturbations Defense
===

This is the official repository for our textual perturbation defense, a simple yet effective method to mitigate backdoor attacks against text-to-image diffusion models.

* 🎯 Text-to-image diffusion models are vulnerable to backdoor attacks, yet the countermeasures remain under-explored.
* ✅ We propose a simple defense! Simply perturbing the input text before feeding it into the text encoder. This is based on the observation that backdoor trigger tokens are typically distant from their original positions after attacks. 
* 📝 Link to the ECCV workshop paper: https://arxiv.org/abs/2408.15721
* 🙌 This project is a collaborative effort between ASUS, National Taiwan University and University of Michigan

Installation
---
Assuming Miniconda is installed, run the following commands to set up the environment:
```bash
conda create -n perturbation python=3.9 -y
conda activate perturbation
pip install -r requirements.txt
```

Usage
---
To use our perturbation framework, please refer to the main function in `src/text_augmenter.py`.

We also provide the chow chow images used in our paper in `data/chow_chow`.

Citation
---
Please consider citing our paper if you find our work helpful. Thank you!
```
@inproceedings{
    chew2024defending,
    title={Defending Text-to-image Diffusion Models: Surprising Efficacy of Textual Perturbations Against Backdoor Attacks},
    author={Oscar Chew and Po-Yi Lu and Jayden Lin and Hsuan-Tien Lin},
    booktitle={ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond},
    year={2024},
    url={https://openreview.net/forum?id=8g2PejwZi6}
}
```
