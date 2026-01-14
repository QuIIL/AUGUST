# AUGUST

## Sequential question–answering AI for hierarchical gastric pathology diagnosis

[Preprint link]() | [Download Model]()| [GastUJB](https://github.com/QuIIL/GastUJB) | [Cite](#reference)

**Abstract:** Artificial intelligence in digital pathology has evolved from task-specific models to multimodal foundation models. However, clinical adoption remains limited by a misalignment between model design and the hierarchical, conditional nature of diagnostic reasoning. In complex domains, such as gastric pathology, diagnosis requires navigating strictly dependent steps where downstream tasks are only clinically valid given specific upstream findings. Regardless, current computational approaches often treat diagnostic tasks in isolation, obscuring these critical dependencies. Herein, we present **AUGUST** (Adaptive Unified Gastric diagnosis Using Sequential Tasks), a sequential question-answering framework designed to mimic the stepwise reasoning of pathologists. **AUGUST** explicitly models diagnostic dependencies by iteratively generating task-specific questions conditioned on prior findings, projecting whole slide images into context-aware representations, and producing answers constrained to clinically valid pathways. We evaluated **AUGUST** on 23,072 gastric whole slide images spanning 19 hierarchical tasks and 79,924 question–answer pairs. In both fully autonomous and human-in-the-loop settings, AUGUST outperformed state-of-the-art multiple instance learning, pathology-specific vision-language, and foundation models on a wide range of datasets and tasks. Our results demonstrate that AUGUST has superior hierarchical consistency and diagnostic accuracy across coarse-, fine-, and grading-levels.

<img src="assets/inference-workflow.png" alt="AUGUST workflow" width="800" />

## What is AUGUST? 
**AUGUST** (**A**daptive **U**nified **G**astric diagnosis **U**sing **S**equential **T**asks) is a sequential question-answering framework designed to mimic the stepwise reasoning of pathologists. It leverages 6,913 whole-slide images (WSIs) from a diverse set of internally collected gastric cases at Catholic University of Korea Uijeongbu St. Mary’s Hospital (2014–2023). Additionally, **AUGUST** explicitly models diagnostic dependencies by iteratively generating task-specific questions conditioned on prior findings, projecting whole slide images into context-aware representations, and producing answers constrained to clinically valid pathways. AUGUST's achieve state-of-the-art performance on a range of datasets and tasks.
- _**Why use AUGUST?**_: Compared to other vision-language, foundation models and multiple instance learning (MIL) that rely on either one of vision-only pretraining or vision-language alignment, AUGUST combined question embbeding and hierarchical workflow into its inner workflow to ensure the gastric cancer diagnosis


<img src="assets/training.png" alt="AUGUST training sessions" width="800" />


## Updates
- **14/01/2026**: AUGUST preprint and dataset [GastUJB](https://github.com/QuIIL/GastUJB) are now live.

## Installation

First clone the repo and cd into the directory:

```bash
git clone https://github.com/QuIIL/AUGUST.git
cd AUGUST
```

Then create a conda env and install the dependencies:

```bash
conda create -n august python=3.10 -y
conda activate august
pip install --upgrade pip
pip install -e .
```

### 1. Getting access

Request access to the model weights from the Huggingface model page [here](https://huggingface.co/khangnq/AUGUST).

### 2. Load model
Following `./notebooks`, AUGUST is loaded by model weights provied by Huggingface. It includes the functionalities to extract slide embeddings from patch embeddings and to perform diagnosis. More details can be found in our demo notebooks.

```python
from august.models.august import AUGUST

model = AUGUST(stage='stage_0') # ['stage_1', 'stage_2','stage_3']
```

### 3. Inference

You can directly use AUGUST for both single-QA and multi-turn QA. AUGUST builds a slide embbeding from a squence of patch embeddings and a question embedding by MIL. Patch setting is 512 x 512 px at magnification is 20x. 

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

conversation = {
    "features_path": "/feature_paths",
        "caption": "location: proximal stomach, body\ncondition: benign tumor, fundic gland polyp",
        "conversation": [
            {
                "question": "Considering the spectrum of gastric pathology, determine the category of diagnosis represented in this slide: inflammatory disease, benign tumor, dysplasia, or cancer.\nPlease format your answer as: 'The category of diagnosis represented in this slide is {inflammatory disease/benign tumor/dysplasia/cancer}.'.",
            }
        ]
}
question = conversation["conversation"][0]
features = h5py.File(conversation["features_path"])

with torch.auto('cude', torch.bfloat16), torch.inference_mode():
    features = h5py.File(conversation["features_path"]).to(device)
    slide_embbeding, question_embedding = model.get_slide_embedding(features, question)
    answer = model(slide_embbeding,question,question_embedding)
```

## Demo for specific use cases

We provide a set of demo notebooks to showcase the capabilities of AUGUST. The notebooks include:
- **Slide embedding extraction** from patch embeddings in `notebooks/slide.ipynb`.
- **Single-QA** on a single slide  in `notebooks/single.ipynb`.
- **Multi-turn QA** evaluation on a single slide dataset in `notebooks/multi-turn.ipynb`.

### Dataset descriptions
- **GastUJB** is a WSIs dataset  was curated from Korea Uijeongbu St. Mary’s Hospital. The dataset can now be accessed [here](https://github.com/QuIIL/GastUJB).



## Reference (!Updating)
If you find our work useful in your research or if you use parts of this code please consider citing our [paper]():



