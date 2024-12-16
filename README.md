<p align="center" width="100%">
<img src="assets/yagna.png" width="20%" alt="RITUAL">
</p>


# 🔥 RITUAL: Random Image Transformations as a Universal Anti-hallucination Lever in Large Vision Language Models

<!-- Arxiv Link, Project Link -->
<div style='display:flex; gap: 0.25rem; '>
<a href="https://arxiv.org/abs/2405.17821"><img src="https://img.shields.io/badge/arXiv-2405.17821-b31b1b.svg"></a>
<a href="https://sangminwoo.github.io/RITUAL"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-blue.svg'></a>
</div>

This repository contains the official pytorch implementation of the paper: "RITUAL: Random Image Transformations as a Universal Anti-hallucination Lever in Large Vision Language Models".


## 🚨 Updates

 * **2024.12.16**: Update Paper / Project page
 * **2024.05.29**: Build project page
 * **2024.05.29**: RITUAL Paper online
 * **2024.05.28**: Code Release


## 👀 Overview
<p align="center" width="100%">
<img src="assets/overview.png" width="70%" alt="Overview">
</p>

**TL;DR:** RITUAL is a simple yet effective anti-hallucination approach for LVLMs. Our RITUAL method leverages basic image transformations (e.g., vertical and horizontal flips) to enhance LVLM accuracy without external models or training. By integrating transformed and original images, RITUAL significantly reduces hallucinations in both discriminative tasks and descriptive tasks. Using both versions together enables the model to refine predictions, reducing errors and boosting correct responses.



## 🤖 RITUAL

<p align="center" width="100%">
<img src="assets/method.png" width="100%" alt="Overview">
</p>

When conditioned on the original image, the probabilities for Blue (correct) and Red (hallucinated) responses are similar, which can lead to the hallucinated response being easily sampled.
RITUAL leverages an additional probability distribution conditioned on the transformed image, where the likelihood of hallucination is significantly reduced.
Consequently, the response is sampled from a linear combination of the two probability distributions, ensuring more accurate and reliable outputs.


## RITUAL+
<p align="center" width="100%">
<img src="assets/ritual+.png" width="65%" alt="Overview">
</p>

In **RITUAL**, the original image V undergoes random transformations, generating a transformed image. In **RITUAL+**, the model evaluates various potential transformations and selects the most beneficial one to improve answer accuracy within the given context, further refining reliability. These transformed images serve as complementary inputs, enabling the model to incorporate multiple visual perspectives to reduce hallucinations.



## 💻 Setup

```bash
conda create -n RITUAL python=3.10
conda activate RITUAL
git clone https://github.com/sangminwoo/RITUAL.git
cd RITUAL
pip install -r requirements.txt
```


## Models

*About model checkpoints preparation*
* [**LLaVA-1.5**](https://github.com/haotian-liu/LLaVA): Download [LLaVA-1.5 merged 7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [**InstructBLIP**](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Download [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)



## 📊 Evaluation

* **POPE**: `bash eval_bench/scripts/pope_eval.sh` 
  - Need to specify "model", "model_path"
* **CHAIR**: `bash eval_bench/scripts/chair_eval.sh`
  - Need to specify "model", "model_path", "type"
* **MME**: `bash experiments/cd_scripts/mme_eval.sh`
  - Need to specify "model", "model_path"

*About datasets preparation*
- Please download and extract the MSCOCO 2014 dataset from [this link](https://cocodataset.org/) to your data path for evaluation.
- For MME evaluation, see [this link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).


## Results

⚠️ All baseline methods were reimplemented within our evaluation setup for fair comparison.

### POPE
<p align="center" width="100%">
<img src="assets/pope.png" width="90%" alt="POPE results">
</p>

### MME
**MME-Fullset**
<p align="center" width="100%">
<img src="assets/mme-fullset.png" width="90%" alt="MME-Fullset results">
</p>

**MME-Hallucination**
<p align="center" width="100%">
<img src="assets/mme-hallucination.png" width="90%" alt="MME-Hallucination results">
</p>

### CHAIR
<p align="center" width="100%">
<img src="assets/chair.png" width="40%" alt="CHAIR results">
</p>



## Examples

<p align="center" width="100%">
<img src="assets/llava_bench.png" width="100%" alt="LLaVA-Bench results">
</p>

<p align="center" width="100%">
<img src="assets/llava_bench_appendix.png" width="100%" alt="LLaVA-Bench results">
</p>



## 🙏 Acknowledgments
This codebase borrows from most notably [VCD](https://github.com/DAMO-NLP-SG/VCD), [OPERA](https://github.com/shikiw/OPERA), and [LLaVA](https://github.com/haotian-liu/LLaVA).
Many thanks to the authors for generously sharing their codes!



## 📝 Citation
If you find this repository helpful for your project, please consider citing our work :)

```
@article{woo2024ritual,
  title={RITUAL: Random Image Transformations as a Universal Anti-hallucination Lever in Large Vision Language Models}, 
  author={Woo, Sangmin and Jang, Jaehyuk and Kim, Donguk and Choi, Yubin and Kim, Changick},
  journal={arXiv preprint arXiv:2405.17821},
  year={2024},
}
```
