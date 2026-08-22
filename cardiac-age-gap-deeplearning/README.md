# Cardiac Biological Age Gap from 12-Lead ECG using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![AHA](https://img.shields.io/badge/AHA-Scientific%20Sessions%202026-red.svg)](https://professional.heart.org/en/meetings/scientific-sessions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Abstract accepted at AHA Scientific Sessions 2026** (Control #2026-A-15429-AHA)  
> *A Deep Learning Approach to Quantifying Cardiac Biological Age Gap from 12-Lead Electrocardiography*

---

## Overview

This project introduces a deep learning framework to compute a **cardiac biological age gap** from 12-lead ECG recordings — the difference between a model's predicted age and the patient's actual chronological age. Inspired by biological age gap paradigms in brain and retinal imaging, we extend this concept to cardiac electrophysiology using the PTB-XL dataset.

### Key Findings

- **Conduction disorders, ST/T-wave changes, and myocardial infarction** show significantly elevated age-adjusted ECG biological age gaps versus normals
- **Females demonstrate consistently larger positive gaps** than males across all pathological categories
- **Lead skewness** (V2: r=0.350, aVL: r=0.342) emerges as the dominant waveform correlate of the age gap — outperforming amplitude-based features

---

## Repository Structure

```
cardiac-age-gap-deeplearning/
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── data_loader.py        # PTB-XL data loading and preprocessing
│   ├── model.py              # 1D ResNet architecture
│   ├── train.py              # Training loop
│   ├── inference.py          # Age gap computation on full dataset
│   ├── age_adjustment.py     # Linear age-adjustment of gap
│   ├── analysis.py           # Statistical analyses (t-tests, Cohen's d)
│   └── waveform_features.py  # Per-lead feature extraction
│
├── notebooks/
│   └── cardiac_age_gap.ipynb # Full end-to-end Colab notebook
│
├── figures/
│   ├── fig1_age_gap_by_category.png
│   ├── fig2_sex_stratified.png
│   └── fig3_waveform_correlations.png
│
├── results/
│   └── summary_statistics.csv
│
└── data/
    └── .gitkeep              # PTB-XL data not included (see below)
```

---

## Methods

### Dataset
- **PTB-XL** (PhysioNet): 21,837 12-lead ECGs, 100Hz, 10-second recordings
- Mean age: 59.8 ± 17.0 years | 52.2% female
- Diagnostic superclasses: NORM (9,522), MI (5,403), STTC (2,807), CD (2,304), HYP (1,309)

### Model
- **Architecture**: 1D Residual Neural Network (ResNet)
- **Input**: Raw 12-lead ECG waveforms (12 × 1000 at 100Hz)
- **Output**: Predicted chronological age (regression)
- **Training**: Exclusively on normal ECGs (n=9,522, 80:20 split)
- **Loss**: L1 Loss (MAE) | **Validation MAE: 10.44 years**

```
Input (12×1000)
    → Conv1d(12→32, k=7) + ReLU
    → ResBlock(32)
    → Conv1d(32→64, stride=2) + ReLU
    → ResBlock(64)
    → Conv1d(64→128, stride=2) + ReLU
    → AdaptiveAvgPool1d(1)
    → Linear(128→64) + ReLU + Dropout(0.3)
    → Linear(64→1)
```

### Age Gap Computation
```
ECG Age Gap = Predicted Age − Chronological Age
Age-Adjusted Gap = Residual after regressing gap on age in normal cohort
```

### Statistical Analysis
- Independent samples t-tests with Cohen's d effect sizes
- Sex-stratified comparisons across all diagnostic categories
- Pearson correlations: per-lead waveform features vs age-adjusted gap

---

## Results

### Age-Adjusted ECG Biological Age Gap by Diagnostic Category

| Category | n | Mean Gap (years) | SD | p-value | Cohen's d |
|---|---|---|---|---|---|
| Normal (ref) | 9,522 | 0.00 | 9.67 | — | — |
| Conduction Disorder | 2,304 | +2.55 | 11.53 | 1.16e-27 | 0.24 |
| ST/T-wave Change | 2,807 | +1.87 | 11.35 | 6.96e-18 | 0.18 |
| Myocardial Infarction | 5,403 | +1.08 | 12.08 | 2.35e-09 | 0.10 |
| Hypertrophy | 1,309 | -0.18 | 11.46 | 0.55 | 0.02 |

### Sex-Stratified Analysis

| Category | Female Gap | Male Gap | p-value |
|---|---|---|---|
| Myocardial Infarction | +2.67 ± 12.00 | -1.61 ± 11.73 | 6.82e-37 |
| ST/T-wave Change | +3.21 ± 10.82 | +0.84 ± 11.64 | 3.95e-08 |
| Conduction Disorder | +3.24 ± 11.56 | +1.55 ± 11.42 | 5.22e-04 |
| Hypertrophy | +0.98 ± 10.90 | -1.70 ± 12.00 | 2.68e-05 |

### Top Waveform Correlates

| Feature | Pearson r | p-value |
|---|---|---|
| V2 Skewness | 0.350 | <0.001 |
| aVL Skewness | 0.342 | <0.001 |
| III Skewness | -0.329 | <0.001 |
| aVF Skewness | -0.279 | <0.001 |
| I Skewness | 0.261 | <0.001 |

---

## Figures

<p align="center">
  <img src="figures/fig1_age_gap_by_category.png" width="600"/>
  <br><em>Figure 1: Age-adjusted ECG biological age gap by diagnostic category</em>
</p>

<p align="center">
  <img src="figures/fig2_sex_stratified.png" width="600"/>
  <br><em>Figure 2: Sex-stratified analysis across diagnostic categories</em>
</p>

<p align="center">
  <img src="figures/fig3_waveform_correlations.png" width="600"/>
  <br><em>Figure 3: Waveform features correlated with cardiac biological age gap</em>
</p>

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cardiac-age-gap-deeplearning.git
cd cardiac-age-gap-deeplearning
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download PTB-XL dataset
```bash
# Option A: via Kaggle (fastest)
kaggle datasets download -d khyeh0719/ptb-xl-dataset --unzip -p ./data

# Option B: via PhysioNet
pip install wfdb
python -c "import wfdb; wfdb.dl_database('ptb-xl/1.0.3', './data')"
```

### 4. Run the full pipeline
```bash
# Train the model
python src/train.py

# Compute age gaps
python src/inference.py

# Run statistical analysis
python src/analysis.py
```

### 5. Or use the Colab notebook
Open `notebooks/cardiac_age_gap.ipynb` directly in Google Colab — no local setup needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/cardiac-age-gap-deeplearning/blob/main/notebooks/cardiac_age_gap.ipynb)

---

## Citation

If you use this code, please cite:

```bibtex
@abstract{paruchuri2026ecg,
  title     = {A Deep Learning Approach to Quantifying Cardiac Biological Age Gap from 12-Lead Electrocardiography},
  author    = {Paruchuri, Amith and Paruchuri, Riddhi},
  booktitle = {AHA Scientific Sessions 2026},
  year      = {2026},
  note      = {Control \#2026-A-15429-AHA}
}
```

---

## Authors

**Amith Paruchuri**  
MBBS Intern, All India Institute of Medical Sciences (AIIMS), New Delhi  
Remote Researcher, Madabhushi Lab, Emory University  
Khorana Scholar, Harvard Medical School  
Co-founder, Aterenex

**Riddhi Paruchuri**  
Ohio State University

---

## Acknowledgements

- PTB-XL dataset: Wagner et al., Nature Scientific Data, 2020
- PhysioNet platform for open-access ECG data
- Inspired by brain age gap (Cole et al.) and retinal age gap (Zhu et al.) frameworks

---

## License

MIT License — see [LICENSE](LICENSE) for details.
