"""
Precompute factor probability distributions for all articles.

Fixes Lfac stuck at log(10)=2.303 by:
1. Searching title AND content (not just title)
2. Title gets 3x weight (more signal-dense)
3. Extended keyword lists for better coverage
4. Soft prior for no-match articles (avoids uniform -> no gradient)

Output: article_factor_labels.npy of shape (N_articles, 10)

Usage:
    python precompute_factor_labels.py
    python precompute_factor_labels.py --articles_path /path/to/articles_max.csv --output_dir /path/to/output/
    python precompute_factor_labels.py --method zero_shot
    python precompute_factor_labels.py --method llm_api
    python precompute_factor_labels.py --method gemini
    python precompute_factor_labels.py --method blend
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Load environment variables from SA/.env file
try:
    from dotenv import load_dotenv
    # pipelines -> v2 -> app -> ai-service -> services -> SA
    # Path: ..\..\..\..\.. from precompute_factor_labels.py
    sa_env = Path(__file__).parent.parent.parent.parent.parent.parent / ".env"
    if sa_env.exists():
        load_dotenv(sa_env)
except ImportError:
    pass  # dotenv not required, will use env vars directly

# ---------------------------------------------------------------------------
# Extended factor keyword ontology (10 classes)
# Base keywords from SAFE-Alert dataset + extended list for better coverage
# ---------------------------------------------------------------------------
FACTOR_KEYWORDS = {
    "institutional_inflow": [
        # base
        "institution", "fund", "grayscale", "microstrategy", "blackrock",
        "corporate", "treasury", "bitcoin purchase", "acquisition", "buy", "accumul",
        # extended
        "pension", "endowment", "sovereign", "wealth fund", "retail", "adoption",
        "purchased", "invested", "allocat",
    ],
    "etf_flow": [
        # base
        "etf", "spot etf", "bitcoin etf", "sec approval", "inflow", "outflow",
        "etf listing", "issuance",
        # extended
        "bitcoin fund", "crypto fund", "futures etf", "bitcoin trust",
        "grayscale premium", "discount", "nav",
    ],
    "regulatory_easing": [
        # base
        "approved", "legal", "regulatory clarity", "compliant", "licensed",
        "framework", "clearance", "authorize",
        # extended
        "regulation", "bill", "congress", "law", "policy", "guidance",
        "exemption", "sandbox", "pilot",
    ],
    "regulatory_tightening": [
        # base
        "ban", "crackdown", "illegal", "sanction", "sec sue", "enforcement",
        "restrict", "prohibited", "suspension",
        # extended
        "warning", "probe", "investigation", "fine", "penalty", "delist",
        "cbdc", "kyc", "aml",
    ],
    "exchange_risk": [
        # base
        "hack", "exploit", "exchange down", "withdrawal halt", "insolvent",
        "ftx", "celsius", "rug", "compromised",
        # extended
        "security", "breach", "stolen", "loss", "bankrupt", "withdrawal",
        "freeze", "halted", "suspicious",
    ],
    "liquidity_squeeze": [
        # base
        "liquidity", "leverage", "liquidation", "margin call", "funding rate",
        "squeeze", "cascade", "deleverag",
        # extended
        "short squeeze", "long squeeze", "open interest", "perp", "futures",
        "basis", "contango", "backwardation",
    ],
    "whale_accumulation": [
        # base
        "whale", "transaction", "on-chain", "address", "wallet", "accumulate",
        "hodl", "large buy", "holdings",
        # extended
        "large transaction", "miner", "mining", "cold storage", "staking",
        "hodler", "accumulation", "supply",
    ],
    "macro_uncertainty": [
        # base
        "inflation", "fed", "interest rate", "recession", "gdp", "cpi",
        "fomc", "yield", "economy", "growth",
        # extended
        "rate hike", "rate cut", "bank", "dollar", "usd", "dxy", "treasury",
        "bonds", "stock market", "equity",
    ],
    "protocol_upgrade": [
        # base
        "upgrade", "fork", "halving", "taproot", "merge", "protocol",
        "layer2", "lightning", "launch", "update",
        # extended
        "mainnet", "testnet", "snapshot", "airdrop", "defi", "nft",
        "smart contract", "validator", "node",
    ],
    "network_outage": [
        # base
        "outage", "congestion", "fees spike", "mempool", "hash rate",
        "51%", "network issue", "downtime",
        # extended
        "slow", "backlog", "stuck", "unconfirmed", "difficulty", "block time",
        "hashpower", "reorg",
    ],
}

FACTOR_NAMES = list(FACTOR_KEYWORDS.keys())
N_FACTORS = len(FACTOR_NAMES)  # 10

# ---------------------------------------------------------------------------
# Source credibility mapping (used for soft prior when no keywords match)
# Financial-focused sources get slight institutional/macro bias.
# ---------------------------------------------------------------------------
# Maps source substring -> factor index weight boost (0-indexed, same as FACTOR_NAMES)
# institutional_inflow=0, etf_flow=1, regulatory_easing=2, regulatory_tightening=3,
# exchange_risk=4, liquidity_squeeze=5, whale_accumulation=6, macro_uncertainty=7,
# protocol_upgrade=8, network_outage=9
SOURCE_PRIOR_BOOSTS = {
    # Financial macro-focused sources: slight macro_uncertainty bias
    "bloomberg":        {7: 0.25},   # macro_uncertainty
    "reuters":          {7: 0.25},
    "wsj":              {7: 0.20},
    "cnbc":             {7: 0.20},
    "financial times":  {7: 0.25},
    "marketwatch":      {7: 0.20},
    # Crypto-native: slight etf_flow + institutional_inflow bias
    "cointelegraph":    {0: 0.15, 1: 0.15},
    "coindesk":         {0: 0.15, 1: 0.15},
    "decrypt":          {0: 0.10, 8: 0.10},  # also tech/protocol
    "bitcoin magazine": {8: 0.20},            # protocol_upgrade
    "newsbtc":          {6: 0.15},            # whale_accumulation
    "coingecko":        {1: 0.10, 6: 0.10},
    # Social/aggregators: slight exchange_risk concern
    "reddit":           {4: 0.10},
    "rss":              {},  # no specific bias
}

LABEL_TEMP = 0.5       # Softmax temperature for keyword-matched articles
                        # T<1 sharpens distribution; 0.5 gives clear peak without
                        # collapsing to one-hot (which breaks cross-entropy gradient).
TITLE_WEIGHT = 3       # Title is 3x more signal-dense than content per word
SOFT_PRIOR_NOISE = 0.02  # Random perturbation added to no-match soft priors


def _score_article(title: str, content: str) -> np.ndarray:
    """
    Compute raw keyword match score for each factor.

    Strategy: weighted keyword count in (title*3 + content)
    - Title treated as 3x-weighted: appended 3 times before joining
    - All keywords are lowercased; matching is substring-based
    - Multi-word keywords (e.g. "interest rate") count as 1 match per occurrence

    Returns:
        scores: (N_FACTORS,) float array of raw match counts (>=0)
    """
    # Normalize text to lowercase; handle None/NaN gracefully
    title_text = str(title).lower() if pd.notna(title) else ""
    content_text = str(content).lower() if pd.notna(content) else ""

    # Concatenate with title repeated TITLE_WEIGHT times for boosted weight
    combined = (title_text + " ") * TITLE_WEIGHT + content_text

    scores = np.zeros(N_FACTORS, dtype=np.float32)
    for j, fname in enumerate(FACTOR_NAMES):
        for kw in FACTOR_KEYWORDS[fname]:
            # Count non-overlapping occurrences
            scores[j] += combined.count(kw.lower())

    return scores


def _softmax_with_temp(scores: np.ndarray, temp: float) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    s = scores / temp
    s -= s.max()  # subtract max for numerical stability
    e = np.exp(s)
    return (e / e.sum()).astype(np.float32)


def _source_prior_base(source: str) -> np.ndarray:
    """Build the deterministic source prior before optional noise."""
    base = np.ones(N_FACTORS, dtype=np.float32) / N_FACTORS
    src_lower = str(source).lower() if pd.notna(source) else ""
    for src_key, boosts in SOURCE_PRIOR_BOOSTS.items():
        if src_key in src_lower:
            for factor_idx, boost_val in boosts.items():
                base[factor_idx] += boost_val
            break
    base /= base.sum()
    return base.astype(np.float32)


def has_source_prior_bias(source: str) -> bool:
    """Return True when a source-specific prior boost is available."""
    src_lower = str(source).lower() if pd.notna(source) else ""
    return any(src_key in src_lower and boosts for src_key, boosts in SOURCE_PRIOR_BOOSTS.items())


def _soft_prior(
    source: str,
    rng: np.random.Generator | None = None,
    noise_scale: float = SOFT_PRIOR_NOISE,
) -> np.ndarray:
    """
    Build a soft prior distribution for articles with no keyword matches.

    When rng is omitted or noise_scale=0, this becomes deterministic. That lets
    live inference reuse the same factor prior logic without injecting runtime
    randomness while training precompute can still add tiny symmetry-breaking
    noise for unmatched articles.
    """
    base = _source_prior_base(source)
    if rng is not None and noise_scale > 0:
        noise = rng.uniform(0.0, noise_scale, size=N_FACTORS).astype(np.float32)
        base = base + noise
        base /= base.sum()
    return base.astype(np.float32)


def compute_factor_probs_for_article(
    title: str,
    content: str,
    source: str = "",
    rng: np.random.Generator | None = None,
    deterministic_prior: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Infer a factor distribution for one article using the shared keyword/source path.

    This helper is reused by both artifact precomputation and live inference so
    the runtime factor path stays closer to the train-time pseudo-label path.
    When keywords match weakly, blend in a small source prior to reduce noisy
    one-factor spikes from sparse keyword hits.
    """
    scores = _score_article(title, content)
    source_prior = _soft_prior(
        source,
        rng=None if deterministic_prior else rng,
        noise_scale=0.0 if deterministic_prior else SOFT_PRIOR_NOISE,
    )

    if scores.sum() <= 0:
        return source_prior.astype(np.float32), False

    score_probs = _softmax_with_temp(scores, LABEL_TEMP)
    strength = float(scores.sum())
    if strength <= 2.0:
        prior_weight = 0.18
    elif strength <= 5.0:
        prior_weight = 0.10
    else:
        prior_weight = 0.05

    probs = (1.0 - prior_weight) * score_probs + prior_weight * source_prior
    probs /= probs.sum()
    return probs.astype(np.float32), True


def compute_factor_labels(
    articles_df: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute factor probability distributions for all articles.

    Args:
        articles_df: DataFrame with columns [title, content, source, timestamp]
        seed: Random seed for reproducible soft priors

    Returns:
        factor_labels: (N_articles, 10) float32 numpy array
            Each row is a probability distribution over 10 factors summing to 1.0
    """
    rng = np.random.default_rng(seed)
    N = len(articles_df)
    factor_labels = np.zeros((N, N_FACTORS), dtype=np.float32)

    # Stats tracking
    n_matched = 0          # articles with at least 1 keyword match
    n_source_biased = 0    # articles that got source-specific prior
    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)  # top factor histogram

    for i, row in enumerate(articles_df.itertuples(index=False)):
        title = getattr(row, "title", "")
        content = getattr(row, "content", "")
        source = getattr(row, "source", "")

        probs, matched = compute_factor_probs_for_article(title, content, source, rng=rng)
        if matched:
            n_matched += 1
        elif has_source_prior_bias(source):
            n_source_biased += 1

        factor_labels[i] = probs

        # Track dominant factor
        factor_top_counts[probs.argmax()] += 1

        if (i + 1) % 1000 == 0:
            pct = (i + 1) / N * 100
            print(f"  Progress: {i+1}/{N} articles ({pct:.1f}%)", flush=True)

    return factor_labels, n_matched, n_source_biased, factor_top_counts


def print_statistics(
    factor_labels: np.ndarray,
    n_matched: int,
    n_source_biased: int,
    factor_top_counts: np.ndarray,
    N: int,
) -> None:
    """Print coverage and distribution statistics."""
    print("\n" + "=" * 60)
    print("FACTOR LABEL STATISTICS")
    print("=" * 60)

    # Coverage: fraction of articles with a dominant factor (max_prob > 0.4)
    max_probs = factor_labels.max(axis=1)
    dominant_mask = max_probs > 0.4
    n_dominant = dominant_mask.sum()
    pct_dominant = n_dominant / N * 100

    print(f"Total articles:          {N}")
    print(f"Keyword-matched:         {n_matched} ({n_matched/N*100:.1f}%)")
    print(f"No match (soft prior):   {N - n_matched} ({(N-n_matched)/N*100:.1f}%)")
    print(f"  - Source-biased prior: {n_source_biased}")
    print(f"  - Near-uniform prior:  {N - n_matched - n_source_biased}")
    print()
    print(f"Dominant factor (max_prob > 0.4): {n_dominant}/{N} ({pct_dominant:.1f}%)")
    print(f"Mean max_prob:           {max_probs.mean():.4f}")
    print(f"Median max_prob:         {np.median(max_probs):.4f}")
    print()

    print("Top-factor distribution:")
    print(f"  {'Factor':<25} {'Count':>8} {'%':>7}")
    print(f"  {'-'*42}")
    for i, fname in enumerate(FACTOR_NAMES):
        count = factor_top_counts[i]
        pct = count / N * 100
        bar = "#" * int(pct / 2)  # scale: 2% per '#'
        print(f"  {fname:<25} {count:>8} {pct:>6.1f}%  {bar}")

    print()
    # Average entropy per article (lower = more confident labels)
    # Uniform distribution has entropy = log(10) = 2.303
    eps = 1e-10
    entropy_per_article = -(factor_labels * np.log(factor_labels + eps)).sum(axis=1)
    print(f"Mean label entropy:      {entropy_per_article.mean():.4f} "
          f"(uniform = {np.log(N_FACTORS):.4f})")
    print(f"Reduction vs uniform:    {(1 - entropy_per_article.mean()/np.log(N_FACTORS))*100:.1f}%")
    print("=" * 60)


def postprocess_factor_labels(
    factor_labels: np.ndarray,
    max_prob_cap: float = 0.97,
    entropy_floor: float = 0.12,
    uniform_mix_max: float = 0.12,
) -> np.ndarray:
    """Soften overly sharp factor distributions to reduce noisy one-hot labels."""
    if factor_labels.size == 0:
        return factor_labels
    num_classes = factor_labels.shape[1]
    uniform = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    def _entropy(p: np.ndarray) -> float:
        p = np.clip(p, 1e-8, 1.0)
        return float(-(p * np.log(p)).sum())

    processed = factor_labels.astype(np.float32).copy()
    for i in range(processed.shape[0]):
        p = processed[i]
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
        max_p = float(p.max())
        ent = _entropy(p)
        if max_p <= max_prob_cap and ent >= entropy_floor:
            processed[i] = p
            continue
        denom = max(max_p - uniform[0], 1e-6)
        mix = min(uniform_mix_max, max(0.0, (max_p - max_prob_cap) / denom))
        p2 = (1.0 - mix) * p + mix * uniform
        p2 = p2 / p2.sum()
        processed[i] = p2.astype(np.float32)
    return processed


# ---------------------------------------------------------------------------
# Human-readable factor label descriptions for zero-shot classification
# ---------------------------------------------------------------------------
FACTOR_LABELS_HUMAN = {
    "institutional_inflow": "institutional investors buying or accumulating Bitcoin",
    "etf_flow": "Bitcoin ETF approval, inflows or outflows",
    "regulatory_easing": "positive regulatory news, approval or legal clarity",
    "regulatory_tightening": "regulatory crackdown, ban or enforcement action",
    "exchange_risk": "exchange hack, insolvency, or security breach",
    "liquidity_squeeze": "leverage liquidation, margin call or liquidity crisis",
    "whale_accumulation": "whale wallet activity or large on-chain transactions",
    "macro_uncertainty": "macroeconomic factors like inflation, Fed policy or recession",
    "protocol_upgrade": "Bitcoin protocol upgrade, halving or network improvement",
    "network_outage": "network congestion, outage or technical issue",
}

# Ordered candidate labels (description strings) aligned with FACTOR_NAMES order
_ZS_CANDIDATE_LABELS = [FACTOR_LABELS_HUMAN[f] for f in FACTOR_NAMES]

# ---------------------------------------------------------------------------
# LLM prompt template for llm_api method
# ---------------------------------------------------------------------------
FACTOR_PROMPT = """Classify this crypto news article into ONE of these 10 financial factors:
1. institutional_inflow - institutions buying/accumulating crypto
2. etf_flow - ETF approvals, inflows, outflows
3. regulatory_easing - positive regulation, approvals, legal clarity
4. regulatory_tightening - bans, crackdowns, enforcement
5. exchange_risk - hacks, insolvency, security breaches
6. liquidity_squeeze - liquidations, margin calls, funding rates
7. whale_accumulation - whale activity, large on-chain transactions
8. macro_uncertainty - Fed, inflation, recession, macro factors
9. protocol_upgrade - halvings, forks, network upgrades
10. network_outage - congestion, downtime, technical issues

Article title: {title}
Article content (first 300 chars): {content}

Respond with ONLY a JSON object: {{"primary": "factor_name", "secondary": "factor_name_or_null", "confidence": 0.0-1.0}}
"""


def _softmax_with_temp_list(scores_list, temp: float = 0.5) -> np.ndarray:
    """Apply softmax with temperature to a list/array of raw scores.

    Returns float32 array of shape (N_FACTORS,) summing to 1.
    """
    arr = np.array(scores_list, dtype=np.float32)
    return _softmax_with_temp(arr, temp)


def compute_factor_labels_zero_shot(
    articles_df: pd.DataFrame,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple:
    """Compute factor labels using BART-MNLI zero-shot classification.

    For each article:
    - Classifies title + content[:512] against 10 human-readable factor descriptions.
    - Applies softmax with T=0.5 to sharpened zero-shot scores.
    - Single-pass (no 2-pass consistency check) for higher coverage & speed.
    - Falls back to keyword method only on rare errors.

    Args:
        articles_df: DataFrame with columns [title, content, source].
        batch_size:  Number of articles per inference batch (default 8).
        seed:        Random seed for reproducible soft priors (fallback path).

    Returns:
        (factor_labels, n_matched, n_source_biased, factor_top_counts)
        where factor_labels is (N_articles, 10) float32 numpy array.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers torch")
        sys.exit(1)

    print("[zero_shot] Loading facebook/bart-large-mnli ...")
    classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    print("[zero_shot] Model loaded. Starting classification ...")

    rng = np.random.default_rng(seed)
    N = len(articles_df)
    factor_labels = np.zeros((N, N_FACTORS), dtype=np.float32)

    n_matched        = 0  # keyword-matched fallbacks
    n_source_biased  = 0
    n_zero_shot_used = 0
    n_fallback       = 0
    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)

    rows = list(articles_df.itertuples(index=False))

    # Process in batches
    for batch_start in range(0, N, batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]
        texts = []
        for row in batch_rows:
            title   = str(getattr(row, "title",   "")).strip()
            content = str(getattr(row, "content", ""))[:512].strip()
            texts.append(title + " " + content)

        # Single pass: single-label classification
        results = classifier(
            texts,
            candidate_labels=_ZS_CANDIDATE_LABELS,
            multi_label=False,
        )

        for local_idx, (row, res) in enumerate(zip(batch_rows, results)):
            global_idx = batch_start + local_idx

            try:
                # Build probability distribution from single-pass scores
                label_to_score = dict(zip(res["labels"], res["scores"]))
                raw_scores = np.array(
                    [label_to_score.get(lbl, 0.0) for lbl in _ZS_CANDIDATE_LABELS],
                    dtype=np.float32,
                )
                probs = _softmax_with_temp(raw_scores, temp=0.5)
                n_zero_shot_used += 1
            except Exception as e:
                # Rare error: fall back to keyword method
                if n_fallback < 5:
                    print(f"    [zero_shot error] {type(e).__name__}: {str(e)[:50]}", flush=True)
                title   = getattr(row, "title",   "")
                content = getattr(row, "content", "")
                source  = getattr(row, "source",  "")
                kw_scores = _score_article(title, content)
                if kw_scores.sum() > 0:
                    probs = _softmax_with_temp(kw_scores, LABEL_TEMP)
                    n_matched += 1
                else:
                    probs = _soft_prior(source, rng)
                    src_lower = str(source).lower() if pd.notna(source) else ""
                    for src_key in SOURCE_PRIOR_BOOSTS:
                        if src_key in src_lower and SOURCE_PRIOR_BOOSTS[src_key]:
                            n_source_biased += 1
                            break
                n_fallback += 1

            factor_labels[global_idx] = probs
            factor_top_counts[probs.argmax()] += 1

        if (batch_start + batch_size) % 100 < batch_size or batch_start + batch_size >= N:
            processed = min(batch_start + batch_size, N)
            print(f"  Progress: {processed}/{N} articles ({processed/N*100:.1f}%) "
                  f"| zero_shot={n_zero_shot_used} fallback={n_fallback}",
                  flush=True)

    print(f"[zero_shot] Done. Zero-shot-classified={n_zero_shot_used}/{N} "
          f"| Fallback-keyword={n_matched} Fallback-prior={n_source_biased}")
    return factor_labels, n_matched, n_source_biased, factor_top_counts


def compute_factor_labels_blend(
    articles_df: pd.DataFrame,
    batch_size: int = 8,
    seed: int = 42,
    zs_conf_threshold: float = 0.7,
) -> tuple:
    """Blend keyword labels with zero-shot when zero-shot is confident.

    Strategy:
      - Always compute keyword labels (fast, deterministic).
      - Compute zero-shot labels once.
      - If zero-shot max_prob >= threshold, use zero-shot; else keep keyword.

    Returns:
        (factor_labels, n_matched, n_source_biased, factor_top_counts)
    """
    print("[blend] Computing keyword labels ...")
    kw_labels, n_matched, n_source_biased, _ = compute_factor_labels(
        articles_df, seed=seed
    )

    print("[blend] Computing zero-shot labels ...")
    zs_labels, _, _, _ = compute_factor_labels_zero_shot(
        articles_df, batch_size=batch_size, seed=seed
    )

    zs_max = zs_labels.max(axis=1)
    use_zs = zs_max >= float(zs_conf_threshold)
    factor_labels = kw_labels.copy()
    factor_labels[use_zs] = zs_labels[use_zs]

    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)
    for p in factor_labels:
        factor_top_counts[int(np.argmax(p))] += 1

    n_zs_used = int(use_zs.sum())
    print(f"[blend] Zero-shot used for {n_zs_used}/{len(articles_df)} "
          f"articles (max_prob >= {zs_conf_threshold:.2f})")

    return factor_labels, n_matched, n_source_biased, factor_top_counts


def compute_factor_labels_llm_api(
    articles_df: pd.DataFrame,
    seed: int = 42,
) -> tuple:
    """Compute factor labels using the Anthropic Claude API.

    For each article:
    - Calls Claude with a structured JSON prompt.
    - Parses primary/secondary factor + confidence from response.
    - Primary factor gets `confidence` weight; secondary gets `1 - confidence`.
    - Consistency filtering: calls API twice for confidence < 0.8; keeps label
      only if both calls agree on primary; otherwise falls back to keyword method.
    - Falls back to keyword method on any API/parse error.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        articles_df: DataFrame with columns [title, content, source].
        seed:        Random seed for reproducible soft priors (fallback path).

    Returns:
        (factor_labels, n_matched, n_source_biased, factor_top_counts)
        where factor_labels is (N_articles, 10) float32 numpy array.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[WARN] ANTHROPIC_API_KEY not set — falling back to zero_shot method.")
        return compute_factor_labels_zero_shot(articles_df, seed=seed)

    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    rng = np.random.default_rng(seed)
    N = len(articles_df)
    factor_labels = np.zeros((N, N_FACTORS), dtype=np.float32)

    n_llm_used      = 0
    n_fallback      = 0
    n_matched       = 0  # keyword fallbacks that had keyword hits
    n_source_biased = 0  # keyword fallbacks that used source prior
    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)

    def _call_api(title: str, content: str):
        """Single API call; returns parsed dict or None on failure."""
        prompt = FACTOR_PROMPT.format(
            title=title,
            content=content[:300],
        )
        try:
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            print(f"    [API error] {exc}", flush=True)
            return None

    def _build_probs_from_api_response(parsed: dict) -> np.ndarray:
        """Convert parsed API JSON to a (N_FACTORS,) probability array."""
        probs = np.zeros(N_FACTORS, dtype=np.float32)
        primary   = parsed.get("primary",   "")
        secondary = parsed.get("secondary", None)
        conf      = float(parsed.get("confidence", 0.7))
        conf      = max(0.0, min(1.0, conf))  # clamp to [0, 1]

        if primary in FACTOR_NAMES:
            p_idx = FACTOR_NAMES.index(primary)
            probs[p_idx] += conf
        else:
            # Unknown primary — distribute confidence evenly
            probs += conf / N_FACTORS

        if secondary and secondary in FACTOR_NAMES and secondary != primary:
            s_idx = FACTOR_NAMES.index(secondary)
            probs[s_idx] += (1.0 - conf)
        else:
            # No valid secondary — give residual weight to primary (or spread)
            if primary in FACTOR_NAMES:
                p_idx = FACTOR_NAMES.index(primary)
                probs[p_idx] += (1.0 - conf)
            else:
                probs += (1.0 - conf) / N_FACTORS

        # Normalize to valid distribution
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(N_FACTORS, dtype=np.float32) / N_FACTORS

        return probs.astype(np.float32)

    def _keyword_fallback(row) -> np.ndarray:
        """Fall back to keyword method for a single article row."""
        nonlocal n_matched, n_source_biased, n_fallback
        title   = getattr(row, "title",   "")
        content = getattr(row, "content", "")
        source  = getattr(row, "source",  "")
        kw_scores = _score_article(title, content)
        n_fallback += 1
        if kw_scores.sum() > 0:
            n_matched += 1
            return _softmax_with_temp(kw_scores, LABEL_TEMP)
        else:
            src_lower = str(source).lower() if pd.notna(source) else ""
            for src_key in SOURCE_PRIOR_BOOSTS:
                if src_key in src_lower and SOURCE_PRIOR_BOOSTS[src_key]:
                    n_source_biased += 1
                    break
            return _soft_prior(source, rng)

    for i, row in enumerate(articles_df.itertuples(index=False)):
        title   = str(getattr(row, "title",   "")).strip()
        content = str(getattr(row, "content", "")).strip()

        parsed1 = _call_api(title, content)
        if parsed1 is None:
            probs = _keyword_fallback(row)
        else:
            conf1 = float(parsed1.get("confidence", 0.7))
            if conf1 < 0.8:
                # Low confidence — call a second time for consistency check
                parsed2 = _call_api(title, content)
                if parsed2 is not None and parsed2.get("primary") == parsed1.get("primary"):
                    # Agreement — use first response
                    probs = _build_probs_from_api_response(parsed1)
                    n_llm_used += 1
                else:
                    # Disagreement or second call failed — fall back to keyword
                    probs = _keyword_fallback(row)
            else:
                # High-confidence single call — use directly
                probs = _build_probs_from_api_response(parsed1)
                n_llm_used += 1

        factor_labels[i] = probs
        factor_top_counts[probs.argmax()] += 1

        if (i + 1) % 50 == 0:
            pct = (i + 1) / N * 100
            print(f"  Progress: {i+1}/{N} articles ({pct:.1f}%) "
                  f"| llm={n_llm_used} fallback={n_fallback}",
                  flush=True)

    print(f"[llm_api] Done. LLM-classified={n_llm_used}/{N} "
          f"| Fallback-keyword={n_matched} Fallback-prior={n_source_biased}")
    return factor_labels, n_matched, n_source_biased, factor_top_counts


def compute_factor_labels_gemini(
    articles_df: pd.DataFrame,
    seed: int = 42,
) -> tuple:
    """Compute factor labels using Google Gemini API (free tier).

    For each article:
    - Calls Gemini with a structured JSON prompt.
    - Parses primary/secondary factor + confidence from response.
    - Single-pass (no 2-pass consistency) to avoid rate limiting.
    - Falls back to keyword method on any API/parse error.

    Requires GOOGLE_API_KEY environment variable.

    Args:
        articles_df: DataFrame with columns [title, content, source].
        seed:        Random seed for reproducible soft priors (fallback path).

    Returns:
        (factor_labels, n_matched, n_source_biased, factor_top_counts)
        where factor_labels is (N_articles, 10) float32 numpy array.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("[WARN] GOOGLE_API_KEY not set — falling back to zero_shot method.")
        return compute_factor_labels_zero_shot(articles_df, seed=seed)

    try:
        import google.generativeai as genai
    except ImportError:
        print("[ERROR] google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")  # Try 2.0-flash first

    rng = np.random.default_rng(seed)
    N = len(articles_df)
    factor_labels = np.zeros((N, N_FACTORS), dtype=np.float32)

    n_gemini_used   = 0
    n_fallback      = 0
    n_matched       = 0
    n_source_biased = 0
    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)

    def _call_gemini_api(title: str, content: str):
        """Single Gemini API call; returns parsed dict or None on failure."""
        import time

        prompt = FACTOR_PROMPT.format(
            title=title,
            content=content[:300],
        )

        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=128,
                    ),
                )
                raw = response.text.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw)
            except Exception as exc:
                # Check if rate limit error
                if "429" in str(exc) or "ResourceExhausted" in str(type(exc).__name__):
                    if attempt < max_retries - 1:
                        # Extract retry delay from API response if available
                        retry_str = str(exc)
                        if "retry in" in retry_str:
                            import re
                            match = re.search(r'retry in ([\d.]+)s', retry_str)
                            if match:
                                retry_delay = float(match.group(1))

                        if n_fallback < 3:  # Only log first few
                            print(f"    [Rate limit] Waiting {retry_delay:.1f}s before retry (attempt {attempt+1}/{max_retries})", flush=True)
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue

                # Print first few errors for other exceptions
                if n_fallback < 5 and attempt == max_retries - 1:
                    print(f"    [API error] {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                return None

    def _build_probs_from_gemini_response(parsed: dict) -> np.ndarray:
        """Convert parsed Gemini JSON to a (N_FACTORS,) probability array."""
        probs = np.zeros(N_FACTORS, dtype=np.float32)
        primary   = parsed.get("primary",   "")
        secondary = parsed.get("secondary", None)
        conf      = float(parsed.get("confidence", 0.7))
        conf      = max(0.0, min(1.0, conf))

        if primary in FACTOR_NAMES:
            p_idx = FACTOR_NAMES.index(primary)
            probs[p_idx] += conf
        else:
            probs += conf / N_FACTORS

        if secondary and secondary in FACTOR_NAMES and secondary != primary:
            s_idx = FACTOR_NAMES.index(secondary)
            probs[s_idx] += (1.0 - conf)
        else:
            if primary in FACTOR_NAMES:
                p_idx = FACTOR_NAMES.index(primary)
                probs[p_idx] += (1.0 - conf)
            else:
                probs += (1.0 - conf) / N_FACTORS

        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(N_FACTORS, dtype=np.float32) / N_FACTORS

        return probs.astype(np.float32)

    def _keyword_fallback(row) -> np.ndarray:
        """Fall back to keyword method for a single article row."""
        nonlocal n_matched, n_source_biased, n_fallback
        title   = getattr(row, "title",   "")
        content = getattr(row, "content", "")
        source  = getattr(row, "source",  "")
        kw_scores = _score_article(title, content)
        n_fallback += 1
        if kw_scores.sum() > 0:
            n_matched += 1
            return _softmax_with_temp(kw_scores, LABEL_TEMP)
        else:
            src_lower = str(source).lower() if pd.notna(source) else ""
            for src_key in SOURCE_PRIOR_BOOSTS:
                if src_key in src_lower and SOURCE_PRIOR_BOOSTS[src_key]:
                    n_source_biased += 1
                    break
            return _soft_prior(source, rng)

    print("[gemini] Starting Gemini API classification ...")
    for i, row in enumerate(articles_df.itertuples(index=False)):
        title   = str(getattr(row, "title",   "")).strip()
        content = str(getattr(row, "content", "")).strip()

        parsed = _call_gemini_api(title, content)
        if parsed is None:
            probs = _keyword_fallback(row)
        else:
            probs = _build_probs_from_gemini_response(parsed)
            n_gemini_used += 1

        factor_labels[i] = probs
        factor_top_counts[probs.argmax()] += 1

        if (i + 1) % 100 == 0:
            pct = (i + 1) / N * 100
            print(f"  Progress: {i+1}/{N} articles ({pct:.1f}%) "
                  f"| gemini={n_gemini_used} fallback={n_fallback}",
                  flush=True)

    print(f"[gemini] Done. Gemini-classified={n_gemini_used}/{N} "
          f"| Fallback-keyword={n_matched} Fallback-prior={n_source_biased}")
    return factor_labels, n_matched, n_source_biased, factor_top_counts


def compute_factor_labels_mistral(
    articles_df: pd.DataFrame,
    seed: int = 42,
) -> tuple:
    """Compute factor labels using Mistral AI API.

    For each article:
    - Calls Mistral with a structured JSON prompt.
    - Parses primary/secondary factor + confidence from response.
    - Single-pass (no 2-pass consistency) for speed.
    - Falls back to keyword method on any API/parse error.

    Requires MISTRAL_API_KEY environment variable.

    Args:
        articles_df: DataFrame with columns [title, content, source].
        seed:        Random seed for reproducible soft priors (fallback path).

    Returns:
        (factor_labels, n_matched, n_source_biased, factor_top_counts)
        where factor_labels is (N_articles, 10) float32 numpy array.
    """
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("[WARN] MISTRAL_API_KEY not set — falling back to zero_shot method.")
        return compute_factor_labels_zero_shot(articles_df, seed=seed)

    try:
        from mistralai.client import Mistral
    except ImportError:
        try:
            from mistralai import Mistral
        except ImportError:
            print("[ERROR] mistralai not installed. Run: pip install mistralai")
            sys.exit(1)

    client = Mistral(api_key=api_key)

    rng = np.random.default_rng(seed)
    N = len(articles_df)
    factor_labels = np.zeros((N, N_FACTORS), dtype=np.float32)

    n_mistral_used  = 0
    n_fallback      = 0
    n_matched       = 0
    n_source_biased = 0
    factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)

    def _call_mistral_api(title: str, content: str):
        """Single Mistral API call; returns parsed dict or None on failure."""
        import time

        prompt = FACTOR_PROMPT.format(
            title=title,
            content=content[:300],
        )

        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                message = client.chat.complete(
                    model="mistral-small",  # Fast, good quality
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=128,
                )
                raw = message.choices[0].message.content.strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw)
            except Exception as exc:
                # Check if rate limit error
                if "429" in str(exc) or "RateLimitError" in str(type(exc).__name__):
                    if attempt < max_retries - 1:
                        if n_fallback < 3:
                            print(f"    [Rate limit] Waiting {retry_delay:.1f}s before retry (attempt {attempt+1}/{max_retries})", flush=True)
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                # Print first few errors
                if n_fallback < 5 and attempt == max_retries - 1:
                    print(f"    [API error] {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                return None

    def _build_probs_from_mistral_response(parsed: dict) -> np.ndarray:
        """Convert parsed Mistral JSON to a (N_FACTORS,) probability array."""
        probs = np.zeros(N_FACTORS, dtype=np.float32)
        primary   = parsed.get("primary",   "")
        secondary = parsed.get("secondary", None)
        conf      = float(parsed.get("confidence", 0.7))
        conf      = max(0.0, min(1.0, conf))

        if primary in FACTOR_NAMES:
            p_idx = FACTOR_NAMES.index(primary)
            probs[p_idx] += conf
        else:
            probs += conf / N_FACTORS

        if secondary and secondary in FACTOR_NAMES and secondary != primary:
            s_idx = FACTOR_NAMES.index(secondary)
            probs[s_idx] += (1.0 - conf)
        else:
            if primary in FACTOR_NAMES:
                p_idx = FACTOR_NAMES.index(primary)
                probs[p_idx] += (1.0 - conf)
            else:
                probs += (1.0 - conf) / N_FACTORS

        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(N_FACTORS, dtype=np.float32) / N_FACTORS

        return probs.astype(np.float32)

    def _keyword_fallback(row) -> np.ndarray:
        """Fall back to keyword method for a single article row."""
        nonlocal n_matched, n_source_biased, n_fallback
        title   = getattr(row, "title",   "")
        content = getattr(row, "content", "")
        source  = getattr(row, "source",  "")
        kw_scores = _score_article(title, content)
        n_fallback += 1
        if kw_scores.sum() > 0:
            n_matched += 1
            return _softmax_with_temp(kw_scores, LABEL_TEMP)
        else:
            src_lower = str(source).lower() if pd.notna(source) else ""
            for src_key in SOURCE_PRIOR_BOOSTS:
                if src_key in src_lower and SOURCE_PRIOR_BOOSTS[src_key]:
                    n_source_biased += 1
                    break
            return _soft_prior(source, rng)

    print("[mistral] Starting Mistral API classification ...")
    for i, row in enumerate(articles_df.itertuples(index=False)):
        title   = str(getattr(row, "title",   "")).strip()
        content = str(getattr(row, "content", "")).strip()

        parsed = _call_mistral_api(title, content)
        if parsed is None:
            probs = _keyword_fallback(row)
        else:
            probs = _build_probs_from_mistral_response(parsed)
            n_mistral_used += 1

        factor_labels[i] = probs
        factor_top_counts[probs.argmax()] += 1

        if (i + 1) % 100 == 0:
            pct = (i + 1) / N * 100
            print(f"  Progress: {i+1}/{N} articles ({pct:.1f}%) "
                  f"| mistral={n_mistral_used} fallback={n_fallback}",
                  flush=True)

    print(f"[mistral] Done. Mistral-classified={n_mistral_used}/{N} "
          f"| Fallback-keyword={n_matched} Fallback-prior={n_source_biased}")
    return factor_labels, n_matched, n_source_biased, factor_top_counts


def main():
    parser = argparse.ArgumentParser(
        description="Precompute factor probability labels for all articles."
    )

    # Resolve default paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    # pipelines/ -> v2/ -> app/ -> ai-service/
    service_root = script_dir.parent.parent.parent
    default_training_data = service_root / "training_data"
    default_training_data_v2 = default_training_data / "v2"

    parser.add_argument(
        "--articles_path",
        type=Path,
        default=None,
        help="Path to articles_max.csv (default: training_data/articles_max.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save article_factor_labels.npy (default: same as articles_path dir)",
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "zero_shot", "llm_api", "gemini", "mistral", "blend"],
        default="keyword",
        help=(
            "Factor label method: "
            "keyword (fast, default), "
            "zero_shot (BART-MNLI), "
            "llm_api (Claude API, requires ANTHROPIC_API_KEY), "
            "gemini (Google Gemini free tier, requires GOOGLE_API_KEY), "
            "mistral (Mistral AI, requires MISTRAL_API_KEY), "
            "blend (zero_shot when confident, else keyword)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for zero_shot inference (default: 8)",
    )
    parser.add_argument(
        "--blend_max_prob",
        type=float,
        default=0.7,
        help="Zero-shot confidence threshold for blend method (default: 0.7).",
    )
    parser.add_argument(
        "--postprocess",
        choices=["off", "auto", "on"],
        default="auto",
        help="Postprocess factor labels to soften overly sharp distributions (default: auto).",
    )
    parser.add_argument(
        "--max_prob_cap",
        type=float,
        default=0.97,
        help="Cap for max factor probability when postprocessing (default: 0.97).",
    )
    parser.add_argument(
        "--entropy_floor",
        type=float,
        default=0.12,
        help="Minimum entropy threshold to avoid overconfident labels (default: 0.12).",
    )
    parser.add_argument(
        "--uniform_mix_max",
        type=float,
        default=0.12,
        help="Maximum uniform-mix ratio when postprocessing (default: 0.12).",
    )
    args = parser.parse_args()

    # Resolve articles path
    if args.articles_path is None:
        base = default_training_data_v2 if default_training_data_v2.exists() else default_training_data
        candidates = [
            base / "articles_max.csv",
            base / "articles.csv",
        ]
        for c in candidates:
            if c.exists():
                args.articles_path = c
                break
        if args.articles_path is None:
            print(f"[ERROR] Could not find articles_max.csv or articles.csv in {base}")
            sys.exit(1)

    if not args.articles_path.exists():
        print(f"[ERROR] Articles file not found: {args.articles_path}")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = args.articles_path.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "article_factor_labels.npy"

    print(f"[START] Precomputing factor labels")
    print(f"  Articles:     {args.articles_path}")
    print(f"  Output:       {output_path}")
    print(f"  Method:       {args.method}")
    print(f"  Factors:      {N_FACTORS} ({', '.join(FACTOR_NAMES)})")
    print(f"  Title weight: {TITLE_WEIGHT}x  (keyword method)")
    print(f"  Temperature:  T={LABEL_TEMP} (softmax sharpening)")
    if args.method == "zero_shot":
        print(f"  Batch size:   {args.batch_size}")
    elif args.method == "llm_api":
        api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        print(f"  ANTHROPIC_API_KEY set: {api_key_set}")
    elif args.method == "gemini":
        api_key_set = bool(os.environ.get("GOOGLE_API_KEY", ""))
        print(f"  GOOGLE_API_KEY set: {api_key_set}")
    elif args.method == "mistral":
        api_key_set = bool(os.environ.get("MISTRAL_API_KEY", ""))
        print(f"  MISTRAL_API_KEY set: {api_key_set}")
    print()

    # Load articles
    print(f"[LOAD] Reading {args.articles_path.name}...")
    articles_df = pd.read_csv(args.articles_path)
    N = len(articles_df)
    print(f"[OK] Loaded {N} articles")

    # Validate required columns
    required_cols = ["title"]
    missing = [c for c in required_cols if c not in articles_df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        print(f"  Available: {list(articles_df.columns)}")
        sys.exit(1)

    # Fill missing optional columns with empty strings
    for col in ["content", "source"]:
        if col not in articles_df.columns:
            print(f"[WARN] Column '{col}' not found - using empty string")
            articles_df[col] = ""

    # Compute factor labels — dispatch to selected method
    if args.method == "keyword":
        print(f"[COMPUTE] Scoring {N} articles (title*{TITLE_WEIGHT} + content)...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = compute_factor_labels(
            articles_df, seed=42
        )
    elif args.method == "zero_shot":
        print(f"[COMPUTE] Zero-shot classification via BART-MNLI "
              f"({N} articles, batch_size={args.batch_size})...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = (
            compute_factor_labels_zero_shot(articles_df, batch_size=args.batch_size, seed=42)
        )
    elif args.method == "blend":
        print(f"[COMPUTE] Blend keyword + zero_shot "
              f"(threshold={args.blend_max_prob:.2f}, batch_size={args.batch_size})...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = (
            compute_factor_labels_blend(
                articles_df,
                batch_size=args.batch_size,
                seed=42,
                zs_conf_threshold=args.blend_max_prob,
            )
        )
    elif args.method == "llm_api":
        print(f"[COMPUTE] LLM-API classification via Anthropic Claude ({N} articles)...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = (
            compute_factor_labels_llm_api(articles_df, seed=42)
        )
    elif args.method == "gemini":
        print(f"[COMPUTE] Gemini API classification ({N} articles)...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = (
            compute_factor_labels_gemini(articles_df, seed=42)
        )
    elif args.method == "mistral":
        print(f"[COMPUTE] Mistral API classification ({N} articles)...")
        factor_labels, n_matched, n_source_biased, factor_top_counts = (
            compute_factor_labels_mistral(articles_df, seed=42)
        )
    else:
        print(f"[ERROR] Unknown method: {args.method}")
        sys.exit(1)

    # Validate output
    assert factor_labels.shape == (N, N_FACTORS), \
        f"Shape mismatch: expected ({N}, {N_FACTORS}), got {factor_labels.shape}"
    assert np.allclose(factor_labels.sum(axis=1), 1.0, atol=1e-5), \
        "Factor labels do not sum to 1.0!"
    assert (factor_labels >= 0).all(), "Negative probabilities detected!"
    assert not np.isnan(factor_labels).any(), "NaN detected in factor labels!"
    print(f"[OK] Shape: {factor_labels.shape}, dtype: {factor_labels.dtype}")

    postprocess_mode = args.postprocess
    postprocess_enabled = (
        postprocess_mode == "on" or
        (postprocess_mode == "auto" and args.method in {"zero_shot", "llm_api", "gemini", "mistral", "blend"})
    )
    if postprocess_enabled:
        print("[POST] Softening factor label distributions...")
        factor_labels = postprocess_factor_labels(
            factor_labels,
            max_prob_cap=args.max_prob_cap,
            entropy_floor=args.entropy_floor,
            uniform_mix_max=args.uniform_mix_max,
        )
        factor_top_counts = np.zeros(N_FACTORS, dtype=np.int64)
        for p in factor_labels:
            factor_top_counts[int(np.argmax(p))] += 1

    # Save
    np.save(output_path, factor_labels)
    print(f"[OK] Saved: {output_path}")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"     File size: {file_size_mb:.2f} MB")

    # Print statistics
    print_statistics(factor_labels, n_matched, n_source_biased, factor_top_counts, N)

    print(f"\n[DONE] Run training with precomputed labels:")
    print(f"  python train_safe_alert.py --symbol BTCUSDT --horizon 1h --epochs 60")
    print(f"  (train_safe_alert.py will auto-load {output_path.name})")


if __name__ == "__main__":
    main()
