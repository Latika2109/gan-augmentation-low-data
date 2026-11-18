# GAN augmentation in low-data image classification

Code, results and figures for a controlled study of whether class-conditional WGAN-GP augmentation helps image classifiers when only 20 to 100 real images per class are available.

**Authors:** Latika Gupta, Prof. Vaibhav Godbole (Fr. Conceicao Rodrigues College of Engineering, Mumbai)

## What was tested

| Item | Setting |
|---|---|
| Datasets | Brain Tumor MRI (4 classes), PlantVillage subset (15 classes) |
| Classifiers | ResNet-18 (all regimes), DenseNet-121 (100 and 20 images per class) |
| Regimes | 100 / 60 / 40 / 20 real images per class (nested subsets) |
| Arms | no augmentation, classical, WGAN-GP at 25/50/100 %, WGAN-GP 50 % + classical |
| Pipelines | `native` and `matched64` (real, synthetic and test images all pass through the same 64 px, 8-bit, bilinear path) |
| Seeds | MRI 0-4, PlantVillage 0-2 (DenseNet-121: seeds 0-2) |
| Scale | 528 classifier runs, 32 GANs |
| Statistics | paired mean difference, 95 % bootstrap CI (10,000 resamples), exact Wilcoxon with Holm correction, Cohen's dz |

GAN: class-conditional WGAN-GP at 64 px, 10,000 critic steps, n_critic = 5, lr 1e-4, betas (0, 0.9), lambda = 10, batch 32. One GAN per (dataset, seed, regime).

## Main findings

- Classical augmentation helps on MRI (paired gain positive in 12 of 12 cells; CI above zero in 8).
- WGAN-GP 50 % does not beat the no-augmentation baseline on MRI with ResNet-18, and 100 % synthetic is worse.
- Adding GAN images on top of classical augmentation is lower than classical alone in 11 of 12 MRI cells.
- The `matched64` control shows the deficit is not just a sharpness shortcut.
- PlantVillage is near ceiling; GAN effects there are suggestive only (three seeds).

Holm-adjusted p-values are all 1.0 at this sample size, so the confidence intervals are the main evidence. With three seeds the bootstrap has few distinct resamples, so PlantVillage intervals are optimistic.

## Repository layout

```
notebooks/Course_Seminar_v3.ipynb   full pipeline (data audit, GANs, classifiers, stats, figures)
results/                            all_runs.csv, summary_accuracy.csv, paired_vs_baseline_*.csv, kid_scores.csv, headline_numbers.json
figures/                            main, gain-over-baseline, ratio-sweep and KID figures
data_reports/                       leakage-audit reports for both datasets
paper/                              (optional) manuscript source
```

## Reproducing

1. Open `notebooks/Course_Seminar_v3.ipynb` in Google Colab with a GPU runtime.
2. Run the cells in order. The notebook mounts Google Drive for checkpoints and downloads both public Kaggle datasets itself:
   - [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
3. The full study is long. Results are cached per run, so an interrupted session resumes where it stopped.

`pip install -r requirements.txt` covers local runs. Datasets and GAN checkpoints are not included.

## Data hygiene

Train and test were audited with MD5 and a 256-bit dHash (Hamming distance <= 4). For MRI, 482 near-duplicate train/test images and 171 within-train duplicates were removed (pool 4,976). For PlantVillage, 7 within-train duplicates were removed (pool 12,780; test 450 images, 30 per class). PlantVillage has several photos of the same leaf, so near-duplicates a hash cannot catch may remain.

Note: the MRI copy used has 1,600 test images, whereas the official Kaggle Testing split has 1,311.

## Implementation note

Augmentation runs in full precision. torchvision `v2.RandomRotation` produced corrupted images inside `torch.autocast`, so `prepare()` disables autocast.

## Citation

```
@misc{gupta2026ganlowdata,
  author = {Gupta, Latika and Godbole, Vaibhav},
  title  = {GAN augmentation in low-data image classification},
  year   = {2026},
  note   = {Code and results: <repository URL>}
}
```

Released under the MIT License.
