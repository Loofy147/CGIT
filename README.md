# CGIT: Cognitive Grammar Induction Test

This repository implements an end-to-end, empirical, runnable **Cognitive Grammar Induction Test (CGIT)** system with robust evolutionary optimization and cross-domain validation, based on the CGIT spec.

## 1. System Architecture

The CGIT system organizes information flow across 4 levels:
*   **Level 4: Cognitive Grammar** — Conditional rules mapping computable *Situation Signatures* to *Operator Sequences* ($S \rightarrow O_1, \dots, O_n$).
*   **Level 3: Operator Sequence** — Actionable programs specifying how observations transform into feature representations.
*   **Level 2: Representation** — Evolved multidimensional array spaces.
*   **Level 1: Data** — Domain-specific inputs and labels.

---

## 2. Core Components

1.  **Primitives (Level 3)**: Operates entirely as domain-agnostic, computable NumPy/scikit-learn transforms:
    *   `DISTINGUISH`: Separates coarse patterns via KMeans clustering, appending contrastive residuals.
    *   `RELATE`: Finds local neighborhood patterns using cosine similarity.
    *   `COMPARE`: Computes distance to global sample mean and nearest neighbors.
    *   `COMPRESS`: Performs PCA to reduce feature dimensionality.
    *   `PREDICT`: Fits linear trends to features to project future patterns.
    *   `ABSTRACT`: Replaces patterns with cluster centroids.
    *   `SPECIALIZE`: Projects specific residuals back to general centroids.
    *   `SIMULATE`: Diffuses information across states via similarity matrix multiplication.

2.  **Situation Encoder (Level 4)**: Maps high-dimensional datasets $(X, y)$ to 7 computable, domain-independent operational signals:
    *   *Noise Level*
    *   *Temporal Dependency*
    *   *Dataset Size / Row Count*
    *   *Contradiction Count*
    *   *Uncertainty / Entropy*
    *   *Feedback Availability*
    *   *Optimization Pressure*

3.  **Fitness Function**: Evaluates a grammar indirectly to prevent complexity bloat and Goodhart failures:
    $$F(G) = \text{Performance}(R_G) - \lambda \cdot \text{Cost}(G) - 0.5 \cdot \text{Fragility}(G)$$

---

## 3. Evolutionary Search & Experimentation

Our evolutionary machinery trains on three source splits and transfers zero-shot to three completely unseen splits:
*   **Training Domains (Source)**: `physics`, `software`, `biology`
*   **Testing Domains (Unseen Target)**: `economics`, `robotics`, `social`

To run the experiment, execute:
```bash
python3 run_experiment.py
```

To run the automated unit-test suite, execute:
```bash
python3 -m unittest test_cgit.py
```

---

## 4. Empirical Outcomes Summary

Our best evolved Grammar achieves outstanding results:
*   **Economics**: Performance ($R^2$ of **0.99859**) with drastically reduced execution cost (**0.03640** vs G2's **0.07280**).
*   **Robotics & Social**: Successfully generalized zero-shot, discovering structured rule-families that outperform hand-crafted static pipelines.
*   **Goodhart Prevention**: Evolved grammar structures remain resilient to distribution shift (added noise) with highly optimal True Fragility bounds.

For a detailed statistical analysis and classification of the evolved grammar families, please refer to [experiment_results.md](experiment_results.md).
