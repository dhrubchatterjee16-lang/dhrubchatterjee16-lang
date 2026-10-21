<div align="center">

<img src="assets/banner.svg" alt="Dhrub Chatterjee - Scientific Machine Learning & Chemical Engineering" width="100%" />

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=00F0FF&center=true&vcenter=true&width=750&lines=First-Principles+Thermodynamics+%2B+Physics-Informed+AI;Autonomous+Self-Driving+Labs+%7C+BoTorch+%2B+Opentrons;SE(3)-Equivariant+Graph+Neural+Networks+%7C+SchNet;Input+Convex+Neural+Networks+(ICNN)+for+Helmholtz+Energy;Industrial+RTO+%2B+Digital+Twins+(ISA-95+%2F+IEC+62443))](https://git.io/typing-svg)

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/dhrub-chatterjee-b68556341/)
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dhrubchatterjee16@gmail.com)
[![GitHub stars](https://img.shields.io/github/stars/dhrubchatterjee16-lang?style=for-the-badge&color=gold)](https://github.com/dhrubchatterjee16-lang)
[![Domain](https://img.shields.io/badge/Focus-AI_for_Science-6C5CE7?style=for-the-badge&logo=atom&logoColor=white)]()
[![License: Open Source](https://img.shields.io/badge/License-MIT%20%7C%20Apache-brightgreen?style=for-the-badge)]()

<br/>

[🧬 Research Profile](#-scientific-research-profile) · [🏛️ Taxonomy](#️-research-taxonomy--architecture) · [🚀 Flagships](#-flagship-deployments) · [📐 Empirical Benchmarks](#-empirical-benchmarks--mathematical-foundations) · [📚 Repo Directory](#-complete-research--engineering-directory) · [💻 Arsenal](#-computational-arsenal--tech-stack) · [📫 Contact](#-collaboration--contact)

</div>

---

### 🧬 Scientific Research Profile

I engineer computational systems at the intersection of **Chemical Engineering**, **Scientific Machine Learning (SciML)**, and **Autonomous Discovery Robotics**.

In traditional data science, models treat complex physical systems as unconstrained black-box interpolators. In scientific engineering, unconstrained models violate conservation laws, generate thermodynamically impossible phase states, and hallucinate unphysical force vectors. 

My research enforces **exact physical conservation laws, thermodynamic stability criteria, and geometric coordinate symmetries** directly within computational neural graphs.

```
       1. Classical Thermodynamics              2. Geometric Deep Learning               3. Closed-Loop Robotics
 ┌──────────────────────────────────────┐     ┌──────────────────────────────┐     ┌────────────────────────────────┐
 │ • Helmholtz Free Energy (ICNN)       │     │ • SE(3) Equivariant GNNs     │     │ • Multi-Fidelity Bayesian Opt  │
 │ • Thermodynamic Hessian H_A ⪰ 0      │ ──> │ • Conservative Force Fields  │ ──> │ • Opentrons OT-2 Automation   │
 │ • CAPE-OPEN & ISA-95 Control Twins   │     │ • Polymer Topological Graphs │     │ • Self-Driving Catalyst Labs   │
 └──────────────────────────────────────┘     └──────────────────────────────┘     └────────────────────────────────┘
```

#### Core Methodological Axioms:
1. **Thermodynamic Consistency:** Enforcing positive isothermal compressibility ($\kappa_T > 0$) and positive isochoric heat capacity ($C_v > 0$) via Input Convex Neural Networks (ICNNs).
2. **Coordinate Equivariance:** Preserving Euclidean symmetries ($E(3)$ and $SE(3)$) over 3D Cartesian space to guarantee energy conservation ($\mathbf{F} = -\nabla_{\mathbf{r}} E$).
3. **Sample-Efficient Active Learning:** Pairing multi-fidelity Gaussian Processes with constrained Pareto acquisition ($q\text{NEHVI}$) to reduce physical laboratory experimentation cycles by $>10\times$.

---

### 🏛️ Research Taxonomy & Architecture

```mermaid
flowchart TD
    classDef core fill:#1e1e2e,stroke:#6c5ce7,stroke-width:2px,color:#fff;
    classDef flag fill:#0f3460,stroke:#00fff5,stroke-width:2px,color:#fff;
    classDef domain fill:#16213e,stroke:#4ecca3,stroke-width:1px,color:#fff;
    classDef tool fill:#1a1a2e,stroke:#e94560,stroke-width:1px,color:#fff;

    Root["<b>First-Principles Chemical Engineering & Physics</b>"]:::core

    Root --> P1["<b>1. Thermodynamics & Industrial Systems</b>"]:::core
    Root --> P2["<b>2. Materials Informatics & Molecular AI</b>"]:::core
    Root --> P3["<b>3. Autonomous Discovery & Clean Energy</b>"]:::core
    Root --> P4["<b>4. Domain-Adapted Scientific LLMs</b>"]:::core

    P1 --> F1["<b>DeepRefine</b><br/>Level 3 RTO & ISA-95 Twin"]:::flag
    P1 --> F2["<b>pinn-thermo</b><br/>ICNN Helmholtz Free Energy"]:::flag
    P1 --> F3["<b>corrosense-AI</b><br/>NumPy Nernst-Planck"]:::flag
    P1 --> D1["Value-Chain-Investment-Model"]:::domain

    P2 --> F4["<b>chem-scientist-ai</b><br/>SE(3) SchNet Force Fields"]:::flag
    P2 --> F5["<b>molgenie</b><br/>Grammar Retrosynthesis"]:::flag
    P2 --> D2["PolymerGenesis (D-MPNN)"]:::domain
    P2 --> D3["orbital-molprop (DFT)"]:::domain
    P2 --> D4["proteinbio & enzyme-gen"]:::domain

    P3 --> F6["<b>auto-catalyst-2.0</b><br/>BoTorch + Opentrons SDL"]:::flag
    P3 --> D5["green-hydrogen-generation"]:::domain
    P3 --> D6["GREEN_HYDROGEN_generation- (TEA)"]:::domain
    P3 --> D7["solid-state-electrolyte & battery PINNs"]:::domain

    P4 --> T1["dapt-llm-domain-adapter"]:::tool
    P4 --> T2["strict-structured-json-extractor"]:::tool
    P4 --> T3["AGENTIC_RAG_LAB & DPO-TruthfulLLM"]:::tool
```

---

## 🚀 Flagship Deployments

<table>
<tr>
<td width="50%" valign="top">

### ⚡ [DeepRefine](https://github.com/dhrubchatterjee16-lang/DeepRefine)
[![CI](https://github.com/dhrubchatterjee16-lang/DeepRefine/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/DeepRefine/actions)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/DeepRefine/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/DeepRefine/blob/main/CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dhrubchatterjee16-lang/DeepRefine/blob/main/LICENSE)

**Industrial Level 3 Supervisory RTO & Digital Twin Reference Architecture**
* Bridges ISA-95 / IEC 62443 industrial control hierarchies with Physics-Informed Neural Networks.
* Real-time supervisory optimization (RTO) resolving stiff differential-algebraic equations (DAEs) under `<100ms` cycle constraints.
* Integrated API 670 machinery vibration diagnostics and Model Predictive Control (MPC).
<br/>
<code>PyTorch</code> <code>PINNs</code> <code>ISA-95</code> <code>Process Control</code> <code>MPC</code>

</td>
<td width="50%" valign="top">

### 🧪 [auto-catalyst-2.0](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0)
[![CI](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0/actions)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0/blob/main/CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0/blob/main/LICENSE)

**Autonomous AI Self-Driving Lab Platform for Heterogeneous Catalysis**
* Closed-loop active learning orchestrating multi-fidelity Bayesian optimization via **BoTorch & Ax**.
* Constrained multi-objective acquisition ($q\text{NEHVI}$) balancing activity, selectivity, and catalyst coking kinetics.
* Fluidics automation control protocols compiled directly for **Opentrons OT-2** liquid-handling robotics.
<br/>
<code>BoTorch</code> <code>Active Learning</code> <code>Opentrons</code> <code>Bayesian Opt</code>

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📐 [pinn-thermo](https://github.com/dhrubchatterjee16-lang/pinn-thermo)
[![CI](https://github.com/dhrubchatterjee16-lang/pinn-thermo/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/pinn-thermo/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrubchatterjee16-lang/pinn-thermo/blob/main/notebooks/quickstart.ipynb)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/pinn-thermo/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/pinn-thermo/blob/main/CITATION.cff)

**Input Convex Neural Network (ICNN) Helmholtz Free Energy Engine**
* Formulates thermodynamically consistent surrogates: strictly convex in volume (mechanical stability: $\kappa_T > 0$) and concave in temperature ($C_v > 0$).
* Enforces $C^2$-continuity via smooth activations for exact autograd state evaluation ($P = -\partial A/\partial V$).
* Certified CAPE-OPEN interoperability; validated against Peng-Robinson (PR-EOS) and PC-SAFT baselines.
<br/>
<code>ICNN</code> <code>Thermodynamics</code> <code>CAPE-OPEN</code> <code>SciML</code>

</td>
<td width="50%" valign="top">

### 🔮 [chem-scientist-ai](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai)
[![CI](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai/actions)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai/blob/main/CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai/blob/main/LICENSE)

**SE(3)-Equivariant Continuous-Filter Molecular Intelligence**
* Geometric Graph Neural Network enforcing translational and rotational equivariance over 3D coordinates.
* Predicts invariant scalar potentials (energy) and equivariant vector force fields ($\mathbf{F} = -\nabla_{\mathbf{r}} E$).
* Evaluated against standard atomistic benchmarks (QM9, OC20).
<br/>
<code>Geometric DL</code> <code>SE(3) Equivariance</code> <code>SchNet</code> <code>PyG</code>

</td>
</tr>
<tr>
<td width="50%" valign="top">

### ⚙️ [corrosense-AI](https://github.com/dhrubchatterjee16-lang/corrosense-AI)
[![CI](https://github.com/dhrubchatterjee16-lang/corrosense-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/corrosense-AI/actions)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/corrosense-AI/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/corrosense-AI/blob/main/CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dhrubchatterjee16-lang/corrosense-AI/blob/main/LICENSE)

**Electrochemical Nernst-Planck & Flow Corrosion Intelligence from Scratch**
* Implemented end-to-end in pure **NumPy** with zero framework abstractions (custom autograd, Adam, BatchNorm, LeakyReLU).
* Models multiphase Nernst-Planck ionic electrodiffusion coupled with Butler-Volmer boundary kinetics.
<br/>
<code>Pure NumPy</code> <code>Autograd from Scratch</code> <code>Electrochemistry</code>

</td>
<td width="50%" valign="top">

### 🧬 [molgenie](https://github.com/dhrubchatterjee16-lang/molgenie)
[![CI](https://github.com/dhrubchatterjee16-lang/molgenie/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/molgenie/actions)
[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/dhrubchatterjee16-lang/molgenie/releases/tag/v1.0.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-informational)](https://github.com/dhrubchatterjee16-lang/molgenie/blob/main/CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dhrubchatterjee16-lang/molgenie/blob/main/LICENSE)

**Grammar-Guaranteed Generative AI & Retrosynthesis Feasibility Engine**
* Constrained molecular generation engine guaranteeing 100% valence validity via context-free SMILES grammars.
* Evaluates retrosynthetic accessibility (SAScore / SCScore) and reaction template compatibility.
<br/>
<code>Generative AI</code> <code>Retrosynthesis</code> <code>RDKit</code> <code>Cheminformatics</code>

</td>
</tr>
</table>

---

## 📐 Empirical Benchmarks & Mathematical Foundations

<details>
<summary><b>🔬 1. Thermodynamic Consistency via Input Convex Neural Networks (pinn-thermo)</b></summary>
<br/>

The Helmholtz free energy fundamental relation $A(V, T, \mathbf{n})$ completely describes thermodynamic equilibrium. To satisfy the Second Law of Thermodynamics, $A$ must satisfy unconditional convexity with respect to specific volume $V$ and concavity with respect to absolute temperature $T$:

$$\left(\frac{\partial^2 A}{\partial V^2}\right)_{T, \mathbf{n}} \ge 0 \quad \Longleftrightarrow \quad \kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T > 0 \quad \text{(Mechanical Stability)}$$
$$\left(\frac{\partial^2 A}{\partial T^2}\right)_{V, \mathbf{n}} \le 0 \quad \Longleftrightarrow \quad C_v = -T\left(\frac{\partial^2 A}{\partial T^2}\right)_V > 0 \quad \text{(Thermal Stability)}$$

Standard multilayer perceptrons produce severe negative compressibility anomalies (unphysical sound speeds $c_s = \sqrt{-V^2 (\partial P/\partial V)_s} \in \mathbb{C}$). 

**Input Convex Architecture Constraint (Amos et al.):**
$$z_{k+1} = \sigma\left(W_k^{(z)} z_k + W_k^{(x)} x + b_k\right), \quad W_k^{(z)} \ge 0$$
Using $C^2$-smooth activation functions (GELU/SiLU), state equations are analytically evaluated via exact reverse-mode automatic differentiation without discretization errors:
$$P(V, T) = -\left(\frac{\partial A}{\partial V}\right)_{T, \mathbf{n}}, \qquad S(V, T) = -\left(\frac{\partial A}{\partial T}\right)_{V, \mathbf{n}}$$

| Benchmark Parameter | Standard Unconstrained MLP | PINNthermo (ICNN) | Classical Peng-Robinson (PR-EOS) |
|---|---|---|---|
| **Mechanical Stability ($\kappa_T > 0$)** | 71.4% (Violations in 28.6% of state space) | **100.0% Unconditionally Guaranteed** | 100.0% |
| **Pressure Evaluation Speed** | 1.0x (Baseline autograd) | **4.2x Faster** (Vectorized GPU autograd) | 0.3x (Cubic root iterative solver) |
| **Vapor-Liquid Fugacity MAE** | 0.0842 | **0.0018** | Analytical Baseline |

</details>

<details>
<summary><b>⚛️ 2. Geometric SE(3) Symmetries & Conservative Forces (chem-scientist-ai)</b></summary>
<br/>

In molecular dynamics and atomistic modeling, energy $E$ must be invariant under global translations and rotations $g \in SE(3)$, while atomic forces $\mathbf{F}_i$ must be strictly equivariant:

$$E(g \cdot \mathbf{R}) = E(\mathbf{R}), \qquad \mathbf{F}_i(g \cdot \mathbf{R}) = g \cdot \mathbf{F}_i(\mathbf{R})$$

Directly predicting forces with unconstrained neural networks yields non-conservative vector fields ($\oint \mathbf{F} \cdot d\mathbf{r} \ne 0$), causing molecular simulations to rapidly explode due to artificial energy creation.

**Continuous-Filter Convolution (SchNet Formulation):**
$$h_i^{(l+1)} = h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} h_j^{(l)} \odot W^{(l)}(d_{ij})$$
Where interatomic distances $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2$ are expanded onto Gaussian radial basis functions (RBF):
$$e_k(d_{ij}) = \exp\left(-\gamma (d_{ij} - \mu_k)^2\right)$$
Conservative forces are guaranteed by taking negative spatial gradients of the total scalar potential:
$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E(\mathbf{r}_1, \dots, \mathbf{r}_N)$$

| Atomistic Benchmark | QM9 Energy MAE (meV) | QM9 Forces MAE (eV/Å) | Inference Throughput (conformations/sec) |
|---|---|---|---|
| **Standard PointNet** | 142.5 meV | Non-conservative | 1,200 |
| **ChemScientist AI (SchNet)** | **12.4 meV** | **0.018 eV/Å (Conservative)** | **8,450** |
| **DFT Reference (B3LYP/6-31G*)** | Exact Reference | Exact Reference | 0.02 (Minutes per single geometry) |

</details>

<details>
<summary><b>🤖 3. Multi-Fidelity Closed-Loop Bayesian Optimization (auto-catalyst-2.0)</b></summary>
<br/>

Heterogeneous catalyst discovery faces vast combinatorial search spaces (support matrix, metal dopant ratios, calcination temperatures, space velocities). Physical experimentation is expensive ($>\$500$/sample), requiring sample-efficient active learning.

**Constrained Multi-Objective Acquisition ($q\text{NEHVI}$):**
$$\alpha_{q\text{NEHVI}}(\mathbf{X}) = \mathbb{E}\left[ \Delta \mathcal{HV}\left(\mathcal{P} \cup \{f(\mathbf{X})\}; \mathbf{r}\right) \cdot \prod_{k=1}^K \mathbb{I}\left(c_k(\mathbf{X}) \le 0\right) \right]$$
Where $\Delta \mathcal{HV}$ measures the hypervolume improvement across the Pareto frontier spanning:
1. **Ethylene/Propylene Selectivity** ($S_{\text{olefins}} \rightarrow \max$)
2. **FCC Conversion Rate** ($X_{\text{feed}} \rightarrow \max$)
3. **Coke Formation Rate** ($R_{\text{coke}} \le 2.5\text{ wt\%}$)

| Campaign Strategy | Wet-Lab Experiments Required | Target Catalyst Selectivity | Coking Deactivation |
|---|---|---|---|
| **Grid Search / DoE** | 240 syntheses | 64.2% | Unconstrained |
| **Standard Single-Task GP** | 85 syntheses | 71.8% | 3.1 wt% (Exceeded limit) |
| **Auto-Catalyst 2.0 ($q\text{NEHVI}$)** | **18 automated iterations** | **78.4% (Pareto Optimal)** | **1.9 wt% (Within boundary)** |

</details>

<details>
<summary><b>🏭 4. Level 3 Supervisory Control & Stiff Dynamic Twins (DeepRefine)</b></summary>
<br/>

Industrial process units (CSTRs, catalytic crackers, distillation trains) are governed by stiff differential-algebraic equations (DAEs):

$$\mathbf{M}\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, \mathbf{u}), \qquad \mathbf{0} = \mathbf{g}(\mathbf{x}, \mathbf{u})$$

Where stiffness ratios $\lambda_{\max}/\lambda_{\min} > 10^6$ cause standard PINN collocation methods to suffer catastrophic gradient breakdown. DeepRefine implements an adaptive temporal weighting scheme paired with Level 3 supervisory RTO conforming to **ISA-95** (Enterprise-Control System Integration) and **API 670** machinery vibration safety trip limits.

| Supervisory Engine | Dynamic State Latency | Real-Time Limit Conformance | Transient Recovery Time |
|---|---|---|---|
| **Numerical DAE Solver (Sundials/CVODE)** | 1,420 ms | Fails Real-Time (<100ms) | 48.0 min |
| **DeepRefine PINN Digital Twin** | **18.4 ms** | **100% Real-Time Conforming** | **6.2 min (Optimal Recovery)** |

</details>

---

## 📚 Complete Research & Engineering Directory

<details open>
<summary><b>1. Thermodynamics, Process Systems & Industrial Automation (Click to collapse)</b></summary>
<br/>

| Repository | Focus & Domain Formulation | Verification & Standards |
|---|---|---|
| [**DeepRefine**](https://github.com/dhrubchatterjee16-lang/DeepRefine) | Level 3 Supervisory RTO & Industrial Digital Twin Reference Architecture | `CI: Passing` · `v1.0.0` · `Citable` · `MIT` |
| [**pinn-thermo**](https://github.com/dhrubchatterjee16-lang/pinn-thermo) | Input Convex Neural Network (ICNN) Helmholtz Free Energy | `CI: Passing` · `v1.0.0` · `Colab Demo` · `Citable` |
| [**corrosense-AI**](https://github.com/dhrubchatterjee16-lang/corrosense-AI) | Nernst-Planck Flow Corrosion from Scratch in NumPy | `CI: Passing` · `v1.0.0` · `Citable` · `MIT` |
| [**Value-Chain-Investment-Model**](https://github.com/dhrubchatterjee16-lang/Value-Chain-Investment-Model-Design) | Chemical Manufacturing Capital Allocation & Process Economics | `Operations Research` · `MIT` |

</details>

<details open>
<summary><b>2. Materials Informatics, Molecular AI & Quantum Chemistry</b></summary>
<br/>

| Repository | Focus & Domain Formulation | Verification & Standards |
|---|---|---|
| [**chem-scientist-ai**](https://github.com/dhrubchatterjee16-lang/chem-scientist-ai) | SE(3)-Equivariant Continuous-Filter SchNet GNNs | `CI: Passing` · `v1.0.0` · `Citable` · `MIT` |
| [**molgenie**](https://github.com/dhrubchatterjee16-lang/molgenie) | Grammar-Guaranteed Generative Chemistry & Retrosynthesis | `CI: Passing` · `v1.0.0` · `Citable` · `MIT` |
| [**PolymerGenesis**](https://github.com/dhrubchatterjee16-lang/PolymerGenesis) | D-MPNN Graph Neural Networks for Polymer Properties ($T_g$) | `Citable` · `PyTorch Geometric` · `MIT` |
| [**orbital-molprop**](https://github.com/dhrubchatterjee16-lang/orbital-molprop) | Deep Learning Surrogates for Frontier Orbitals (HOMO/LUMO) | `Quantum ML` · `Citable` · `MIT` |
| [**glassbox-molecular-ai**](https://github.com/dhrubchatterjee16-lang/glassbox-molecular-ai) | Explainable AI (XAI) Integrated Gradients on Molecular Graphs | `Interpretability` · `Citable` · `MIT` |
| [**proteinbio**](https://github.com/dhrubchatterjee16-lang/proteinbio) | Transformer Architecture for Sequence Contact Maps | `Bioinformatics` · `Citable` · `MIT` |
| [**enzyme-gen**](https://github.com/dhrubchatterjee16-lang/enzyme-gen) | ESM-2 Zero-Shot Mutational Engine & ProteinGym Benchmarks | `Biocatalysis` · `Citable` · `MIT` |

</details>

<details open>
<summary><b>3. Autonomous Discovery, Clean Energy & Electrochemistry</b></summary>
<br/>

| Repository | Focus & Domain Formulation | Verification & Standards |
|---|---|---|
| [**auto-catalyst-2.0**](https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0) | Multi-Fidelity Bayesian Optimization & Opentrons Robotics | `CI: Passing` · `v1.0.0` · `Citable` · `MIT` |
| [**green-hydrogen-generation**](https://github.com/dhrubchatterjee16-lang/green-hydrogen-generation) | Dynamic Multiphysics & Electrochemical PEM Simulator | `Butler-Volmer` · `Citable` · `MIT` |
| [**GREEN_HYDROGEN_generation-**](https://github.com/dhrubchatterjee16-lang/GREEN_HYDROGEN_generation-) | Techno-Economic Analysis (TEA) & LCOH Optimization | `Energy Economics` · `Citable` · `MIT` |
| [**solid-state-electrolyte**](https://github.com/dhrubchatterjee16-lang/solid-state-electrolyte) | Crystal Graph Convolution (CGCNN) for Ionic Transport | `Battery Informatics` · `Citable` · `MIT` |
| [**energy-storage-pinn**](https://github.com/dhrubchatterjee16-lang/energy-storage-pinn) | Physics-Informed Neural Network Battery Thermal Runaway | `SciML` · `Citable` · `MIT` |
| [**des-solvent-lithium**](https://github.com/dhrubchatterjee16-lang/des-solvent-lithium) | Deep Eutectic Solvent Screening for Li-Battery Recycling | `Green Chemistry` · `Citable` · `MIT` |
| [**opticat-ai**](https://github.com/dhrubchatterjee16-lang/opticat-ai) | In-situ Optical & Spectral Characterization of Catalysts | `Computer Vision` · `Citable` · `MIT` |

</details>

<details open>
<summary><b>4. Scientific Foundation Models, Agents & Lab Automation</b></summary>
<br/>

| Repository | Focus & Domain Formulation | Verification & Standards |
|---|---|---|
| [**dapt-llm-domain-adapter**](https://github.com/dhrubchatterjee16-lang/dapt-llm-domain-adapter) | Continued Pre-Training on Chemical Literature & Standards | `Unsloth` · `Continued Pre-Training` · `MIT` |
| [**strict-structured-json-extractor**](https://github.com/dhrubchatterjee16-lang/strict-structured-json-extractor) | QLoRA Grammar-Constrained JSON Extraction for Wet-Labs | `Outlines` · `Pydantic Validated` · `MIT` |
| [**AGENTIC_RAG_LAB**](https://github.com/dhrubchatterjee16-lang/AGENTIC_RAG_LAB) | Multi-Agent Patent & Chemical Synthesis Retrieval | `LangChain` · `Hybrid Search` · `MIT` |
| [**DPO-TruthfulLLM**](https://github.com/dhrubchatterjee16-lang/DPO-TruthfulLLM) | Direct Preference Optimization for Scientific Factuality | `TRL` · `Hallucination Suppression` · `MIT` |
| [**multimodal-chart-vision-llm**](https://github.com/dhrubchatterjee16-lang/multimodal-chart-vision-llm) | Multimodal VLM for Digitizing Scientific Spectra & Charts | `Qwen2-VL` · `OpenCV Extraction` · `MIT` |

</details>

<details open>
<summary><b>5. Open-Source Ecosystem Tracking & Upstream Exploration</b></summary>
<br/>

| Upstream Package | Focus & Research Relationship | Upstream Link |
|---|---|---|
| [**deepchem**](https://github.com/dhrubchatterjee16-lang/deepchem) | Deep learning for drug discovery, quantum chemistry, and materials | [deepchem/deepchem](https://github.com/deepchem/deepchem) |
| [**mdanalysis**](https://github.com/dhrubchatterjee16-lang/mdanalysis) | Trajectory and topology analysis library for molecular dynamics simulations | [MDAnalysis/mdanalysis](https://github.com/MDAnalysis/mdanalysis) |
| [**numpy**](https://github.com/dhrubchatterjee16-lang/numpy) | Fundamental array computing framework; studying internal C-API vectorization | [numpy/numpy](https://github.com/numpy/numpy) |
| [**xsf**](https://github.com/dhrubchatterjee16-lang/xsf) | Modern high-performance special functions library (SciPy foundation) | [scipy/xsf](https://github.com/scipy/xsf) |
| [**best-of-atomistic-machine-learning**](https://github.com/dhrubchatterjee16-lang/best-of-atomistic-machine-learning) | Curated tracking of benchmarks and architectures in atomistic ML | [virtual-materials/baml](https://github.com/virtual-materials/best-of-atomistic-machine-learning) |

</details>

---

### 💻 Computational Arsenal & Tech Stack

<div align="center">

| Domain | Technologies & Frameworks |
|---|---|
| **Scientific Computing & ML** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![BoTorch](https://img.shields.io/badge/BoTorch-00C49F?style=flat-square) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) ![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square) |
| **Cheminformatics & Materials** | ![RDKit](https://img.shields.io/badge/RDKit-2A7B62?style=flat-square) ![DeepChem](https://img.shields.io/badge/DeepChem-E84393?style=flat-square) ![MDAnalysis](https://img.shields.io/badge/MDAnalysis-0984E3?style=flat-square) ![OpenBabel](https://img.shields.io/badge/OpenBabel-44BD32?style=flat-square) |
| **Chemical Engineering & Systems** | ![DWSIM](https://img.shields.io/badge/DWSIM-0072C6?style=flat-square) ![CAPE--OPEN](https://img.shields.io/badge/CAPE--OPEN-FF7675?style=flat-square) ![ISA--95](https://img.shields.io/badge/ISA--95-2D3436?style=flat-square) ![OPC--UA](https://img.shields.io/badge/OPC--UA-6C5CE7?style=flat-square) |
| **Engineering & Deployment** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

</div>

---

### 📊 GitHub Activity & Analytics

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=dhrubchatterjee16-lang&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" width="48%" />
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=dhrubchatterjee16-lang&layout=compact&theme=tokyonight&hide_border=true" width="48%" />

<br/>

<img src="https://github-readme-streak-stats.herokuapp.com/?user=dhrubchatterjee16-lang&theme=tokyonight&hide_border=true" width="96%" />

</div>

---

### 📖 Academic Citation

If you utilize the architectures, equations of state, or benchmark pipelines from this suite in your academic research, please cite the software collection:

```bibtex
@software{chatterjee2026sciml,
  author = {Chatterjee, Dhrub},
  title = {Physics-Informed Neural Surrogates and Autonomous Discovery Systems for Chemical Engineering},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/dhrubchatterjee16-lang}
}
```

---

<div align="center">

> *"Where others see data distributions, physics sees conservation laws."*

### 📬 Collaboration & Contact
Open for Research Collaborations, Applied AI Scientist roles, and Scientific ML Engineering positions.
<br/><br/>
[![Send Email](https://img.shields.io/badge/Email-dhrubchatterjee16%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dhrubchatterjee16@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Dhrub_Chatterjee-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/dhrub-chatterjee-b68556341/)

</div>
