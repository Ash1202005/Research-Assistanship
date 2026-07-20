# Research Assistantship: Enhancing Early-Exiting Frameworks for Long-Tailed Image Classification

This repository contains the implementation, experiments, reports, and presentation developed during my research assistantship on long-tailed image classification. The work focuses on reproducing state-of-the-art methods and designing novel hybrid architectures that combine **ELF (Early-Exiting Framework)** and **ResLT (Residual Learning for Long-Tailed Recognition)**.

## Overview

Long-tailed image classification remains a challenging problem due to severe class imbalance. Existing approaches primarily address this problem from two different perspectives:

- **ELF** improves computational efficiency by allowing easy samples to exit early during inference.
- **ResLT** improves recognition of minority classes using residual learning and specialized classifier heads.

This research investigates whether combining these complementary ideas can produce a more accurate and efficient framework for long-tailed recognition.

---

## Research Objectives

- Reproduce the **ELF** framework from scratch.
- Reproduce the **ResLT** framework from scratch.
- Design and evaluate hybrid architectures combining both methods.
- Study the effects of different loss functions, routing strategies, and data augmentation techniques.
- Identify stable optimization strategies for hybrid early-exiting models.

---

## Research Workflow

### Phase 1 — ELF Implementation

Implemented the **Early-Exiting Framework (ELF)** from scratch on the CIFAR-10-LT benchmark, reproducing the original architecture, training strategy, and evaluation pipeline.
- Implemented ELF Paper using a ResNet-32 backbone.
- Tuned hyperparameters across CIFAR-10-LT datasets (10×, 50×, 100× imbalance).
- Reproduced results within a small margin of the original paper.
**Detailed implementation:** [`ELF Implementation.pdf`](./ELF%20Implementation.pdf)

---

### Phase 2 — ResLT Implementation

Implemented **ResLT (Residual Learning for Long-Tailed Recognition)** from scratch using a ResNet-32 backbone. Performed extensive hyperparameter tuning, data augmentation, and optimization to closely reproduce the performance reported in the original paper.
- Implemented ResLT using a ResNet-32 backbone.
- Tuned hyperparameters across CIFAR-10-LT datasets (10×, 50×, 100× imbalance).
- Applied AutoAugmentation and Label Smoothing.
- Reproduced results within a small margin of the original paper.

**Detailed implementation:** [`ResLT on CIFAR-10-LT.pdf`](./ResLT%20on%20CIFAR-10-LT.pdf)

---

### Phase 3 — Hybrid Architecture Design

Designed and evaluated multiple hybrid architectures combining the strengths of **ELF** and **ResLT**. The study explores different routing strategies, loss formulations, optimization techniques, and augmentation methods to identify effective architectures for long-tailed image classification.

**Hybrid Implementation:** [`Hybrid Report.pdf`](./Hybrid%20Report.pdf)

---

### Phase 4 — Experimental Analysis

Systematically evaluated multiple hybrid models by comparing:

- Loss accumulation vs. routed loss
- Cross-Entropy vs. LDAM
- AutoAugment vs. Mixup
- Different routing thresholds
- Performance across 10×, 50×, and 100× imbalance settings

More than **16 hybrid architectures** were implemented, trained, and benchmarked.

## Key Findings

- Loss accumulation across exits significantly destabilizes optimization.
- Routing each sample to a single loss produces considerably more stable training.
- Cross Entropy with Label Smoothing consistently outperformed LDAM within the hybrid framework.
- Mixup further improved routed hybrid models.
- The proposed hybrid architectures achieved performance comparable to, and in some settings slightly exceeds  (+0.26%) , the original state-of-the-art implementations.

---

## Dataset

- **CIFAR-10-LT**
- Long-tailed versions with:
  - 10× imbalance
  - 50× imbalance
  - 100× imbalance

---

## Technologies

- Python
- PyTorch
- NumPy
- Torchvision
- Google Colab

---

## Research Contributions

- Reproduced two state-of-the-art long-tailed learning frameworks.
- Designed and evaluated 16 hybrid ELF–ResLT architectures.
- Investigated multiple optimization strategies for early-exiting networks.
- Demonstrated that **loss routing without accumulation** is a significantly more stable optimization strategy for hybrid early-exiting architectures.
- Demonstrated that routing samples to a single loss without accumulation produced the most stable hybrid architecture, achieving
79.59% (100×), 82.89% (50×), and 89.32% (10×) accuracy, outperforming all internally developed baselines and slightly
exceeding (+0.26%) the original ResLT implementation on the 10× benchmark.
---
