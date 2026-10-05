# MedCred-Bench: Reproducibility Package and Statutory Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Reproducibility: Deterministic](https://img.shields.io/badge/Reproducibility-Verified%20(100%25)-brightgreen.svg)]()
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)]()

Official open-source evaluation benchmark, Description Logic ontology schemas, and reproducibility audit suite for the research paper:

> **Mitigating Generative Hallucinations in Compliance-Critical Medical Manufacturing Recruitment: A Tri-Agent Verification Architecture with Semantic Entropy Calibration**  
> *Status: Under Peer Review at ICDTM 2026 (ACM ICPS)*  
> *Contact: junjun ([3588@duck.com](mailto:3588@duck.com))*

---

## ⚖️ Statutory Data Governance & Privacy Statement

Pursuant to the **Personal Information Protection Law of the People's Republic of China (PIPL, Articles 13 & 28)**, the **European Union General Data Protection Regulation (EU GDPR, Article 9)**, and corporate Non-Disclosure Agreements (NDAs):

* **Raw Proprietary Resumes**: Historical job applicant resumes collected from medical device manufacturers contain Personally Identifiable Information (PII) and are legally confidential. They cannot be released publicly on open web repositories.
* **Standardized Synthetic Benchmark**: This repository provides **`MedCred-Bench`**, an open-access, fully de-identified synthetic benchmark comprising **1,200 fine-grained credential verification tasks across 250 candidate dossiers**. It precisely models the statistical distributions, statutory job requirements, and adversarial failure modes observed in real-world regulatory recruitment pipelines.
* **De-identified Enterprise Access**: Academic researchers requiring access to raw de-identified enterprise records may submit a formal Data Use Agreement (DUA) request to the corresponding author.

---

## ⚡ 10-Second Quickstart: Deterministic Table 2 Metric Audit

To independently audit the **Table 2** confusion matrices, accuracy rates, and 95% Wilson Score / Bootstrap confidence intervals reported in the manuscript, run the zero-dependency verification script (requires standard Python 3.8+ with no third-party libraries):

```bash
python replicate_table2_metrics.py
```

### Expected Output Summary:
* **Autonomous Processing Subset ($N = 783$, 81.6% Coverage)**:
  * Contingency Counts: $\text{TP} = 547, \; \text{FP} = 3, \; \text{TN} = 231, \; \text{FN} = 2$ ($\sum = 783$)
  * Precision: $99.45\%$ (95% Wilson CI: $[98.4\%, 99.8\%]$)
  * Recall: $99.64\%$ (95% Wilson CI: $[98.7\%, 99.9\%]$)
  * $\text{F}_1$ (positive class): $99.55\%$ (95% Bootstrap CI: $[99.1\%, 99.9\%]$)
  * Residual Hallucination Rate: $0.38\% \approx 0.4\%$ (95% Wilson CI: $[0.1\%, 1.1\%]$)
* **Full Pipeline with HITL Adjudication ($N = 960$, 100% Coverage)**:
  * Contingency Counts: $\text{TP} = 484, \; \text{FP} = 13, \; \text{TN} = 436, \; \text{FN} = 27$ ($\sum = 960$)
  * Precision: $97.4\%$ (95% Wilson CI: $[95.6\%, 98.5\%]$)
  * Recall: $94.7\% \approx 94.8\%$ (95% Wilson CI: $[92.4\%, 96.3\%]$)
  * $\text{F}_1$ (positive class): $96.0\% \approx 96.1\%$ (95% Bootstrap CI: $[94.7\%, 97.2\%]$)
  * Residual Hallucination Rate: $0.4\%$ (95% Wilson CI: $[0.2\%, 1.1\%]$)
  * Human Escalation Rate (HITL): $18.4\%$ ($177/960$, 95% Wilson CI: $[16.1\%, 21.0\%]$)

---

## 📂 Repository Structure

```text
medcred-bench/
├── README.md                      # Comprehensive documentation and reproducibility guide
├── LICENSE                        # MIT Open-Source License
├── requirements.txt               # Optional environment dependencies
├── replicate_table2_metrics.py    # Zero-dependency instant verification auditor
├── simulate_tri_agent_uq.py       # Full Tri-Agent simulation, baselines, and ablation engine
├── data/
│   ├── medcred_bench_1200_tasks.json     # All 1,200 fine-grained credential tasks
│   ├── benchmark_split_ids.json          # Strict 20%/80% dossier-level partition IDs
│   └── sample_qualitative_traces.json    # 3 representative adversarial case traces
└── ontology/
    └── statutory_dl_schemas.json         # Description Logic (ALCQIO) axioms & registrars
```

---

## 🔬 Benchmark Specification (`MedCred-Bench`)

`MedCred-Bench` models multi-agent verification under high-stakes medical device compliance requirements (e.g., ISO 13485:2016, ISO 14644 cleanroom, China NMPA Decree No. 739, U.S. FDA 21 CFR Part 820 QMSR).

### 1. Benchmark Composition (1,200 Tasks across 250 Dossiers)
* **200 Authentic-Profile Dossiers (1,000 Tasks)**:
  * **Track 1: Cleanroom GMP & Sterilization** (50 dossiers, 250 tasks)
  * **Track 2: ISO 13485 Quality Assurance & V&V** (50 dossiers, 250 tasks)
  * **Track 3: Class II / Class III Regulatory Affairs** (50 dossiers, 250 tasks)
  * **Track 4: Medical Electronics & Biocompatibility** (50 dossiers, 250 tasks)
* **50 Adversarial Stress Dossiers (200 Tasks)**:
  * **Type I: Fabricated Certification Injection** (15 dossiers, 60 tasks)
  * **Type II: Spatiotemporal Collision & Overlap** (15 dossiers, 60 tasks)
  * **Type III: Scope-of-Practice Distortion** (10 dossiers, 40 tasks)
  * **Type IV: Adversarial Injection & Subtle Boundary Tampering** (10 dossiers, 40 tasks)

### 2. Zero-Leakage 20%/80% Dossier-Level Split
Partitioning is enforced strictly at the **dossier level** to avoid cross-task contamination:
* **Calibration Split (20%)**: 50 dossiers (240 tasks: 40 authentic + 10 adversarial) dedicated to calibrating the epistemic safety gating threshold $\tau_{\text{safe}} = 0.35$.
* **Held-Out Evaluation Split (80%)**: 200 dossiers (960 tasks: 160 authentic + 40 adversarial) held out exclusively for unbiased evaluation.

---

## 🧠 Tri-Agent Architecture Overview

```text
Raw Candidate Dossier (D)
        │
        ▼
[Stage 1: Extractor Agent (A_ext)]  ──► Grammar-constrained context-free decoding (JSON)
        │
        ▼
[Stage 2: Deterministic Symbolic Checks]
        ├── Ontological Verifier (A_ver)  ──► Description Logic (K_statute) predicate check
        └── Adversarial Auditor (A_aud)   ──► Spatiotemporal conflict graph (G_conflict)
        │
        ├── Any hard failure? ──► [Deterministic Rejection (Reject)]
        │
        ▼ (Deterministic Pass)
[Stage 3: Epistemic Uncertainty Quantification]
        ├── Parallel stochastic generation (M = 5 reasoning completions at T = 0.7)
        ├── DeBERTa-v3-large-MNLI bidirectional entailment clustering
        └── Shannon Semantic Entropy SE(D, q) computation
        │
        ├── If SE > tau_safe (0.35) ──► [Human-in-the-Loop Cockpit (Route_HITL)]
        ├── If SE <= 0.35 and Majority = 1 ──► [Autonomous Approval (Approve)]
        └── If SE <= 0.35 and Majority = 0 ──► [Autonomous Rejection (Reject)]
```

---

## 🚀 Running Full End-to-End Simulations

To run the complete benchmark generation, threshold calibration, baseline comparisons (Single LLM CoT, Ensemble Majority, Dual-Agent, Tri-Agent), and ablation experiments:

```bash
# Optional: install dependencies
pip install -r requirements.txt

# Run full simulation (generates empirical report)
python simulate_tri_agent_uq.py
```

---

## 📜 Citation

If you find `MedCred-Bench` or the Tri-Agent Verification Architecture helpful in your research, please cite our paper:

```bibtex
@misc{medcred2026mitigating,
  title={Mitigating Generative Hallucinations in Compliance-Critical Medical Manufacturing Recruitment: A Tri-Agent Verification Architecture with Semantic Entropy Calibration},
  author={MedCred-Bench Contributors},
  year={2026},
  note={Under peer review at ICDTM 2026 (ACM ICPS)}
}
```

---

## 📬 Contact & Support

For inquiries regarding benchmark access, Description Logic schemas, or evaluation protocols:
* **Contact**: junjun
* **Email**: [3588@duck.com](mailto:3588@duck.com)
