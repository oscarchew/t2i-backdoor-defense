Textual Perturbations Defense
===

This is the official repository for our textual perturbation defense, a simple yet effective method to mitigate backdoor attacks against text-to-image diffusion models.

* 🎯 Text-to-image diffusion models are vulnerable to backdoor attacks, yet the countermeasures remain under-explored.
* ✅ We propose a simple defense! Simply perturb the input text before feeding it into the text encoder. This is based on the observation that backdoor trigger tokens are typically distant from their original positions after attacks.
    * TextualPerturbation: simple perturbations based on random character/word manipulations
    * PEPPER: perception-guided perturbation
* 📝 Link to the [EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.548/) and [ECCV 2024 workshop](https://arxiv.org/abs/2408.15721) papers.
* 🙌 This project is a collaborative effort between Texas A&M University, ASUS, National Taiwan University and University of Michigan

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
To use our perturbation framework, please refer to `src/pepper.py` and `src/textual_perturbation.py`

We also provide the chow chow images used in our 2024 paper in `data/chow_chow`.

Citation
---
Please consider citing our paper if you find our work helpful. Thank you!
```bibtex
@inproceedings{chew-etal-2026-pepper,
    title = "{PEPPER}: Perception-Guided Perturbation for Robust Backdoor Defense in Text-to-Image Diffusion Models",
    author = "Chew, Oscar and Lu, Po-Yi  and Lin, Jayden  and Huang, Kuan-Hao  and Lin, Hsuan-Tien",
    editor = "Lee, Hung-Yi and Dev, Sunipa and Li, Jessy and Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.548/",
    pages = "11562--11574",
    ISBN = "979-8-89176-462-0",
}
```

```bibtex
@inproceedings{
    chew2024defending,
    title={Defending Text-to-image Diffusion Models: Surprising Efficacy of Textual Perturbations Against Backdoor Attacks},
    author={Oscar Chew and Po-Yi Lu and Jayden Lin and Hsuan-Tien Lin},
    booktitle={ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond},
    year={2024},
    url={https://openreview.net/forum?id=8g2PejwZi6}
}
```
