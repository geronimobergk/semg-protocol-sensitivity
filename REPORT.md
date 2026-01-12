# Evaluation Protocols Matter - A Controlled Study on Protocol Sensitivity in  sEMG CNNs

## 1. Motivation & Problem Statement

Performance comparisons in surface electromyography (sEMG)–based gesture recognition are notoriously difficult to interpret. Reported accuracies vary widely across publications, often without a clear distinction between *what type of generalization* is being evaluated. In particular, subject-dependent and cross-subject evaluation protocols are frequently used interchangeably, despite probing fundamentally different capabilities.

This ambiguity is problematic for two reasons. First, sEMG signals exhibit strong inter-subject variability, making the choice of evaluation protocol a dominant factor in reported performance. Second, architectural conclusions—such as the benefit of attention mechanisms—are often drawn from a single protocol, without assessing whether these conclusions remain valid under distribution shift to unseen users.

**Objective.**
The goal of this project is not to introduce a new model, but to *systematically analyze how supervised evaluation protocols influence both absolute performance and model ranking*, under tightly controlled conditions. By fixing architecture, training budget, optimization settings, and efficiency constraints, this study isolates the effect of the evaluation protocol itself.

**Definition (Protocol Sensitivity).**
In this work, protocol sensitivity is defined as the change in performance and/or model ranking induced solely by a change in evaluation protocol, under otherwise identical experimental conditions.

This project is intended as a methodological foundation for subsequent work on representation learning and label-efficient adaptation in sEMG systems.

---

## 2. Experimental Setup

### Dataset and Scope

* **Dataset:** NinaPro DB2
* **Subjects:** 40
* **Signals:** sEMG only
* **Gesture subset:** Exercise B
* **Sampling rate:** 2 kHz

The scope is intentionally narrow in order to reduce confounding factors and ensure interpretability of protocol-induced effects.

### Windowing and Preprocessing

* **Window length:** 200 ms
* **Hop size:** 10 ms

Raw sEMG signals are used directly, without handcrafted features.
Because heavy window overlap can introduce information leakage, all dataset splits are performed at the **repetition level**, ensuring that no overlapping signal segments appear in both training and test sets.

### Models

Two ** spatio-temporal CNN architectures** are evaluated:

* **ST-CNN:** baseline spatio-temporal convolutional network
* **ST-Attn-CNN:** identical backbone augmented with spatio-temporal attention

Both models share:

* ~2.6M parameters
* ~71.6M MACs per inference window
* ~143.2M FLOPs per inference window

### Training Protocol

All experiments use:

* Identical optimizer and learning rate schedule
* Identical number of training epochs (fixed budget)
* No protocol-specific hyperparameter tuning

While protocol-specific tuning could improve absolute performance, it would confound attribution. The objective here is not to optimize each protocol independently, but to assess *robustness and ranking stability under fixed inductive and optimization assumptions*.

### Evaluation Protocols

Three supervised evaluation protocols are considered, each probing a distinct capability:

1. **Pooled Subject-Dependent, Repetition-Disjoint**

   * Training and testing on all 40 subjects
   * Train/test split by repetition labels
   * Measures generalization across repetitions for *seen users*

2. **Cross-Subject (Leave-One-Subject-Out, LOSO)**

   * Training on 39 subjects
   * Testing on one unseen subject
   * Repeated for all subjects (full LOSO)
   * Measures out-of-the-box generalization to new users

3. **Single-Subject, Repetition-Disjoint**

   * Training and testing on a single subject
   * Serves as a personalization ceiling and reference point

For all protocols, validation data are drawn exclusively from the training partition and are repetition-disjoint from both training and test sets. In the LOSO setting, the held-out test subject is never used for validation or model selection; evaluation is performed on a fixed repetition index to ensure strict comparability across protocols.

| Protocol                     | Train subjects | Val subjects | Test subjects | Split axis             |
| ---------------------------- | -------------- | ------------ | ------------- | ---------------------- |
| `single_subject_repdisjoint` | 1 subject      | same subject | same subject  | repetitions            |
| `pooled_repdisjoint`         | all subjects   | all subjects | all subjects  | repetitions            |
| `loso`                       | N−1 subjects   | N−1 subjects | 1 subject     | subjects + repetitions |

---

## 3. Results

### 3.1 Performance Across Protocols

Table 1 reports **Balanced Accuracy (BA)** and **Macro-F1** for both models under the three evaluation protocols. Results are averaged across subjects (LOSO), repetitions, and random seeds (I, however, ran the experiment with a single seed only to reduce computation cost.). Standard deviations reflect inter-subject variability.

#### Table 1 – Performance across evaluation protocols (mean ± std)

| Protocol                     | Model       | Balanced Acc. [%] | Macro-F1 [%] |
| ---------------------------- | ----------- | ----------------- | ------------ |
| Single-subject, rep-disjoint | ST-CNN      | **73.8 ± 6.4**    | 73.5 ± 6.6   |
|                              | ST-Attn-CNN | 63.5 ± 7.7        | 63.2 ± 8.0   |
| Pooled, rep-disjoint         | ST-CNN      | **65.1 ± 0.0**    | 65.2 ± 0.0   |
|                              | ST-Attn-CNN | 65.0 ± 0.0        | 65.1 ± 0.0   |
| Cross-subject (LOSO)         | ST-CNN      | **26.4 ± 7.4**    | 23.7 ± 7.6   |
|                              | ST-Attn-CNN | 26.4 ± 6.9        | 23.8 ± 7.3   |

Across both architectures, **absolute performance is dominated by the evaluation protocol**:

* **Single-subject evaluation** yields the highest performance, reflecting a personalization ceiling in which models can exploit subject-specific signal structure and are not required to resolve inter-user variability.
* **Pooled repetition-disjoint evaluation** results in intermediate performance, revealing the increased task complexity of learning a shared decision boundary across multiple users, even though the model is not required to generalize to unseen subjects at test time.
  The absence of observed variance in this setting (± 0.0) arises because the experiment was executed with a single seed only. Thus there was only a single run for this protocol.
* **Cross-subject LOSO evaluation** leads to a dramatic performance drop of nearly 40 percentage points in balanced accuracy, highlighting the difficulty of out-of-the-box generalization to unseen users.

In the **single-subject setting**, ST-CNN outperforms the attention-augmented variant. This suggests that, under strong personalization and limited distribution shift, the simpler convolutional architecture provides a more effective inductive bias for within-user repetition generalization. **This observation does not imply that attention mechanisms are ineffective in general**, but rather that their benefits do not manifest under this tightly controlled experiment with a fixed training budget of only 50 epochs.

Importantly, these large performance differences arise **despite identical architectures, training budgets, optimization settings, and preprocessing**, confirming that **evaluation protocol choice is a first-order determinant of reported performance in sEMG classification**.

---

### 3.2 Protocol-Dependent Model Comparison

Beyond absolute performance, the results allow comparison of the two architectures under different generalization regimes.

| Evaluation Protocol                 | Primary Generalization Axis | Better Model | Δ Balanced Accuracy (pp) | Interpretation    |
| ----------------------------------- | --------------------------- | ------------ | ------------------------ | ----------------- |
| Single-subject, repetition-disjoint | Repetitions (within-user)   | ST-CNN       | +10.3                    | Clear advantage   |
| Pooled, repetition-disjoint         | Repetitions (seen users)    | None         | +0.1                     | Indistinguishable |
| Cross-subject (LOSO)                | Subjects (unseen users)     | None         | 0.0                      | Indistinguishable |

In the **single-subject setting**, ST-CNN substantially outperforms the attention-augmented variant, suggesting that the simpler architecture provides a stronger inductive bias for within-user repetition generalization.

In contrast, under **pooled repetition-disjoint** and **cross-subject LOSO** evaluation, the two models exhibit **nearly identical performance**, with differences well below one percentage point. Within the observed variance, the architectures are statistically indistinguishable.

Crucially, this indicates that **architectural differences that matter in a personalized setting largely vanish once subject variability becomes the dominant source of distribution shift**.

---

### 3.3 Quantifying Protocol Sensitivity

Relative drops normalize for absolute performance differences and directly quantify how sensitive a model is to changes in the generalization regime. For two protocols (A \rightarrow B), protocol sensitivity is defined as:

$
\Delta_{\text{rel}}(A \rightarrow B) = \frac{\text{BA}_{A} - \text{BA}_{B}}{\text{BA}_{A}}
$

Three transitions are considered, each isolating a distinct source of generalization difficulty:

* **Single-subject → Pooled repetition-disjoint**
  (removal of personalization; increased inter-user variability without subject shift at test time)
* **Pooled repetition-disjoint → Cross-subject (LOSO)**
  (introduction of subject shift to unseen users)
* **Single-subject → Cross-subject (LOSO)**
  (combined effect of removing personalization and introducing subject shift)

#### Table 3 – Protocol sensitivity (relative BA drop)

| Model       | Single → Pooled [%] | Pooled → LOSO [%] | Single → LOSO [%] |
| ----------- | ------------------- | ----------------- | ----------------- |
| ST-CNN      | 11.8                | **59.4**          | **64.2**          |
| ST-Attn-CNN | −2.4                | **59.4**          | **58.4**          |

Several conclusions follow directly from these normalized results.

First, **subject shift is the dominant driver of protocol sensitivity**. Both architectures exhibit an identical relative performance drop of approximately **59%** when moving from pooled subject-dependent evaluation to cross-subject LOSO, indicating that generalization to unseen users overwhelms both architectures.

Second, **personalization effects are comparatively small and architecture-dependent**. The transition from single-subject to pooled evaluation leads to a modest drop for ST-CNN (11.8%), while ST-Attn-CNN shows no degradation, highlighting that within-subject advantages do not translate consistently across architectures.

Third, the **end-to-end deployment gap is substantial**. Relative drops of **58–64%** from single-subject to cross-subject evaluation quantify the discrepancy between optimistic personalized performance and realistic out-of-the-box deployment scenarios.

Overall, these results demonstrate that **protocol sensitivity is both large and structured**: while architectural choices influence subject-dependent performance, **cross-subject evaluation induces a consistent and severe performance collapse across models**, reinforcing evaluation protocol choice as a first-order concern in sEMG research.

---

## 4. Discussion

### 4.1 Interpretation

The results lead to a clear and methodologically important conclusion:
**evaluation protocol choice dominates both absolute performance and apparent architectural effects in  sEMG CNNs**.

While the two architectures differ meaningfully in the single-subject setting, these differences collapse under pooled and cross-subject evaluation. In the LOSO regime, where models are tested on entirely unseen users, performance is uniformly low and nearly identical across architectures.

This suggests that **architectural refinements alone are insufficient to address the core challenge of cross-subject generalization in sEMG systems**, at least under fixed training budgets and without protocol-specific tuning. Gains observed under subject-dependent protocols do not reliably translate to out-of-the-box deployment scenarios.

Importantly, these conclusions **could not be reached from a single evaluation protocol alone**. If only single-subject or pooled repetition-disjoint results were reported, one might incorrectly infer meaningful architectural advantages that do not persist under subject shift.

---

### 4.2 Implications for Evaluation Practice

These findings highlight a critical risk in the sEMG literature:
**architectural conclusions drawn from subject-dependent evaluation protocols may not generalize to realistic deployment settings**.

Because subject identity induces strong latent structure in sEMG signals, evaluation protocols that do not explicitly control for subject leakage can substantially overestimate real-world performance and obscure generalization limits.

From a system-design perspective, this underscores the need for:

* explicit alignment between evaluation protocol and deployment scenario,
* careful separation of repetition- and subject-level generalization,
* protocol-aware reporting of results and conclusions.

---

### 4.3 Limitations and Scope

This study is intentionally constrained and its conclusions should be interpreted accordingly:

* The evaluated models are server-scale  CNNs, not TinyML architectures, and the results do not directly translate to ultra-low-power, fully on-device deployments.
* Only a single random seed is used per experiment to limit computational cost, which may underestimate variance due to initialization effects. Since protocol-induced effects exceed 50% relative drops, they however dominate initialization variance by an order of magnitude, making single-seed comparisons sufficient for this report.
* Training is conducted with a fixed and relatively short budget of 50 epochs, potentially limiting convergence and favoring architectures with faster early learning dynamics.
* Only one dataset (NinaPro DB2) and a single exercise subset (Exercise B) are considered, limiting generalization across datasets, sensor configurations, and gesture taxonomies.
* No subject calibration, domain adaptation, or personalization mechanisms are applied in the cross-subject setting, even though such steps are common in practical systems.
* No protocol-specific hyperparameter tuning or adaptation is performed.
* Only supervised learning is evaluated.

These limitations are deliberate design choices to isolate protocol-induced effects under tightly controlled conditions. Consequently, the results are comparative and diagnostic rather than performance-optimal, and should not be interpreted as upper bounds on achievable accuracy.

---

### 4.4 Conclusion and Outlook

**This study shows that evaluation protocol choice alone can reverse architectural conclusions in  sEMG CNNs, highlighting protocol sensitivity as a first-order concern for both scientific interpretation and real-world deployment.**

The observed collapse of architectural differences under cross-subject evaluation motivates future work beyond supervised model design, including:

* cross-subject evaluation of self-supervised and pretraining-based representation learning algorithms,
* label-efficient personalization and rapid user adaptation,
* protocol-aware evaluation frameworks that jointly assess accuracy, robustness, and efficiency.

While demonstrated on sEMG, the underlying issue of protocol sensitivity is likely pervasive in other biosignal and wearable sensing domains characterized by strong inter-user variability.
