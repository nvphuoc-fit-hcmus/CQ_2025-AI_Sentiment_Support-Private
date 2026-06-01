# SAFE-Alert Paper — Erratum

Two equations as printed in the paper are mathematically degenerate and do
not match the training behavior reported in the thesis. The code implements
the corrected versions (see [DEVIATIONS.md](DEVIATIONS.md) L4 and L5); the
paper text must be updated to match. This document provides the exact
substitutions.

---

## Erratum #1 — Equation 34 (Selection loss L_sel)

### As printed
$$
\mathcal{L}_{sel} \;=\; \Big(\sum_{i=1}^{N} \tilde\alpha_i \;-\; K_h\Big)^2
                      \;+\; \eta \sum_{i=1}^{N} \tilde\alpha_i \log \tilde\alpha_i
$$

### Problem
After the top-K hard mask, $\tilde\alpha_i$ is a **probability simplex**
(softmax output). Therefore
$$
\sum_i \tilde\alpha_i \;=\; 1 \quad \text{(or exactly } K_h \text{ under scaled-softmax)} \;\;\; \forall \text{ samples},
$$
which means the cardinality term $\bigl(\sum_i \tilde\alpha_i - K_h\bigr)^2$
is **constant w.r.t. the network parameters** and has **zero gradient**.
The term is decorative: it cannot teach the selector to pick exactly $K_h$
articles because the quantity it penalizes does not depend on anything the
selector can change.

### Corrected form
Replace the cardinality operand with the **pre-softmax sigmoid gates**
$\sigma(a_i)$, where $a_i$ is the raw selection score before the top-K
softmax:
$$
\boxed{
\mathcal{L}_{sel} \;=\; \Big(\sum_{i=1}^{N} \sigma(a_i) \;-\; K_h\Big)^2
                      \;+\; \eta \sum_{i=1}^{N} \tilde\alpha_i \log \tilde\alpha_i
}
$$
Now $\sum_i \sigma(a_i) \in [0, N]$ varies freely per sample, so the
regularizer honestly pulls the expected number of selected articles toward
$K_h$. The entropy term retains the simplex-normalized $\tilde\alpha_i$
(unchanged).

### Implementation reference
[`safe_alert_training_utils.py:422-454`](pipelines/safe_alert_training_utils.py) — note that `alpha_raw = σ(scores)` is separate
from `alpha_tilde = softmax · topK` which is still used for the entropy
term.

---

## Erratum #2 — Equation 36 (Faithfulness loss L_faith)

### As printed
$$
\mathcal{L}_{faith} \;=\; \max\Big(0,\; m \;-\; \bigl(\hat p_{full} \;-\; \hat p_{mask}\bigr)\Big)
$$
with a **fixed absolute margin** $m$.

### Problem
The absolute gap $\hat p_{full} - \hat p_{mask}$ is **bounded by
$\hat p_{full}$**:
* If the model is confident ($\hat p_{full} = 0.9$), the maximum achievable
  gap is $0.9$ — margin $m=0.12$ is trivially satisfied.
* If the model is uncertain ($\hat p_{full} = 0.4$), the maximum possible
  gap is $0.4$ — margin $m=0.12$ is 30 % of the maximum and very hard to
  achieve even when the selected articles are genuinely important.

The loss therefore concentrates gradient on easy (already-confident)
samples and gives vanishing signal on hard samples, which is the opposite
of what a faithfulness regularizer should do.

### Corrected form — relative gap with per-horizon margin
Normalize the gap by $\hat p_{full}$ and retune margins per horizon:
$$
\boxed{
\mathcal{L}_{faith} \;=\; \max\Big(0,\; m_h \;-\; \frac{\hat p_{full} \;-\; \hat p_{mask}}{\hat p_{full}}\Big)
}
$$

**Per-horizon margins $m_h$** (retuned for the relative scale):

| Horizon $h$ | $m_h$ |
|-------------|-------|
| 15 m        | 0.08  |
| 1 h         | 0.12  |
| 4 h         | 0.15  |
| 24 h        | 0.18  |

The relative gap is scale-free: a model that drops from 0.9 → 0.78 and one
that drops from 0.4 → 0.35 both yield the same $\sim 0.13$ relative gap,
equalising gradient pressure across confidence regimes.

### Implementation reference
[`safe_alert_training_utils.py:355-394`](pipelines/safe_alert_training_utils.py) — `rel_gap = (p_full - p_mask) / max(p_full, ε)`.

---

## Erratum #3 — Section 4.5 Hyperparameter table (addition)

The paper's §4.5 implementation section omits several non-default
regularisers that are necessary for the reported numbers to reproduce.
Please append the following table to §4.5.4:

**2026-05-04 audit note.** Treat the table below as a paper-text erratum for
the historical implementation, not as the live source of truth for every
experiment. New runs must cite the saved YAML artifact. The frozen comparison
configs are:

| Config | Purpose | Key values |
|--------|---------|------------|
| `train_config_research_best.yaml` | Current research baseline | `balanced_batch_sampler=true`, `policy_confidence_source=raw`, `lambda4=0.0`, `lambda6=0.0`, `label_smoothing_eps=0.0`, `focal_gamma=0.3`, `class_weights_mode=balanced`, `entropy_anchor_weight=0.0`, `logit_l2_weight=0.0`, `epsilon_h_override=null`; historical 1h-only value `0.0015` is commented in YAML |
| `train_config_paper_strict.yaml` | Paper-literal comparison canary | `balanced_batch_sampler=false`, `policy_confidence_source=raw`, `lambda4=0.2`, `lambda6=0.10`, `epsilon_h_override=null` |

| Component                  | Value                     | Source                          |
|---------------------------|--------------------------|---------------------------------|
| Dropout (article enc.)     | 0.20                     | [safe_alert_net.py:128](models/safe_alert_net.py) |
| Dropout (fusion MLP)       | 0.10                     | —                               |
| Dropout (head input)       | 0.35                     | [safe_alert_net.py:554](models/safe_alert_net.py) |
| Label smoothing $\epsilon$ | 0.05                     | L1                              |
| Focal $\gamma$             | 1.0                      | L1                              |
| Class weights              | sqrt-softened balanced   | L1, M5                          |
| Entropy anchor weight      | 0.05                     | L9                              |
| Entropy anchor target      | $0.50 \ln C$             | L9                              |
| Logit L2 weight            | $1\!\cdot\!10^{-4}$      | L8                              |
| STE top-K                  | enabled                  | A1                              |
| Factor dropout             | 0.10                     | A5                              |
| Temperature bounds         | $[0.5, 2.5]$             | M1                              |
| SWA start fraction         | 0.88                     | M4                              |
| ES patience (Stage 3)      | 4 epochs                 | T5                              |
| Factor-pathway grad scale  | 0.1                      | A2, M6                          |

---

## Summary of changes required to paper text

1. **§3.9.4 / Eq.34** — replace the cardinality operand $\sum_i \tilde\alpha_i$
   with $\sum_i \sigma(a_i)$. Entropy term unchanged. Add one sentence
   explaining the simplex-constant degeneracy of the original formulation.

2. **§3.9.6 / Eq.36** — replace the absolute gap with the relative gap
   $(\hat p_{full} - \hat p_{mask}) / \hat p_{full}$. Add the per-horizon
   margin table.

3. **§4.5.4** — append the hyperparameter table above and add one sentence
   that run-specific values are audited from the saved YAML artifact.

No change is needed to any other equation, figure, or claim. The four
contributions, the architecture diagram, and the experimental protocol
remain as described.
