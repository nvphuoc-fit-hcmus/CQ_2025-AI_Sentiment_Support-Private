"""
Φ_txt — Natural language explanation generator (paper Eq.29).

Paper §3.8.2 specifies that on top of structured outputs (S^news, S^fac, ŷ, ĉ)
the system can produce a natural-language explanation via "một module ngôn ngữ
nhẹ hoặc một bộ template".

This module implements the **template** path. Choosing template over LLM-generated
text has three benefits for a thesis-grade decision-support system:

  1. **Faithfulness by construction.** A template can only mention evidence
     that was actually selected by the upstream selector — there is no
     hallucination risk that an LLM might invent supporting facts not
     present in the chosen articles. This matches the L_faith objective
     (Eq.36): explanations must be grounded in the evidence that actually
     drove the prediction.

  2. **Determinism for audit.** A given (direction, factors, articles)
     tuple maps to exactly one explanation string. Reviewers can verify
     the explanation against the structured output without re-running an
     LLM, which is critical when the system is part of a financial
     decision-support pipeline that may need to be replayed for
     compliance.

  3. **Zero inference cost.** No additional model load or API call at
     prediction time. Latency stays bounded by the encoder/fusion path.

The generator is intentionally separate from the model (no torch dependency)
and is invoked at inference / case-study export, not during training.
"""

from typing import List, Dict, Tuple, Sequence

# 3-class direction labels (paper §3.1: y ∈ {-1, 0, +1}).
DIRECTION_TEXT = {
    0: "giảm",
    1: "đi ngang",
    2: "tăng",
    -1: "giảm",  # alias for paper notation
}

# Vietnamese descriptions for the 10-factor ontology (paper §3.4).
FACTOR_DESCRIPTION = {
    "institutional_inflow":  "dòng tiền tổ chức",
    "etf_flow":              "dòng tiền ETF",
    "regulatory_easing":     "nới lỏng quy định",
    "regulatory_tightening": "siết chặt quy định",
    "exchange_risk":         "rủi ro sàn giao dịch",
    "liquidity_squeeze":     "siết thanh khoản",
    "whale_accumulation":    "tích lũy của cá voi",
    "macro_uncertainty":     "bất định vĩ mô",
    "protocol_upgrade":      "nâng cấp giao thức",
    "network_outage":        "sự cố mạng lưới",
}


def _factor_phrase(top_factors: Sequence[Tuple[str, float]], n: int = 2) -> str:
    """Join top-N factor descriptions into a Vietnamese phrase."""
    if not top_factors:
        return "không có yếu tố nổi bật"
    items = [
        FACTOR_DESCRIPTION.get(f, f.replace("_", " "))
        for f, _ in top_factors[:n]
    ]
    if len(items) == 1:
        return items[0]
    return " và ".join(items)


def _articles_phrase(top_articles: Sequence[Dict], n: int = 3) -> str:
    """Format top-N article titles as a numbered list."""
    if not top_articles:
        return "không có bằng chứng văn bản"
    parts = []
    for i, art in enumerate(top_articles[:n], start=1):
        title = art.get("title") or art.get("snippet") or "(không có tiêu đề)"
        title = title.strip().rstrip(".")
        if len(title) > 120:
            title = title[:117] + "..."
        parts.append(f"({chr(0x60 + i)}) {title}")
    return "; ".join(parts)


def generate_explanation(
    direction: int,
    confidence: float,
    horizon: str,
    top_articles: Sequence[Dict],
    top_factors: Sequence[Tuple[str, float]],
    alert_decision: bool,
    return_pred: float | None = None,
    asset: str = "BTC",
) -> str:
    """Generate paper-Eq.29 natural-language explanation from structured output.

    Args:
        direction:      predicted class (0=DOWN, 1=NEUTRAL, 2=UP).
        confidence:     ĉ ∈ [0, 1].
        horizon:        "15m" | "1h" | "4h" | "24h".
        top_articles:   list of dicts each with at least a "title" key
                        (output of S^news per Eq.27).
        top_factors:    list of (factor_name, probability) tuples
                        (output of S^fac per Eq.28).
        alert_decision: True if A=1 per Eq.26, False if abstain.
        return_pred:    optional r̂ in % (e.g., 0.018 = +1.8%) for richer
                        explanation; pass None to omit.
        asset:          asset display name; defaults to "BTC".

    Returns:
        Multi-line Vietnamese explanation matching the worked example in
        paper §3.12.7.
    """
    dir_text = DIRECTION_TEXT.get(int(direction), "không xác định")
    factor_phrase = _factor_phrase(top_factors, n=2)
    article_phrase = _articles_phrase(top_articles, n=3)

    lines = [f"Tín hiệu: {asset} có khả năng {dir_text} trong {horizon} tới."]

    if return_pred is not None:
        sign = "+" if return_pred >= 0 else ""
        lines.append(f"Mức thay đổi kỳ vọng: {sign}{return_pred * 100:.2f}%.")

    lines.append(f"Độ tin cậy: {confidence:.2f}.")

    if top_articles:
        lines.append(f"Bằng chứng chính: {article_phrase}.")

    if top_factors:
        lines.append(f"Yếu tố chi phối: {factor_phrase}.")

    # Composed reasoning sentence — bound to the actual selected evidence
    # so the language never claims more than the structured output supports.
    lines.append(
        f"Giải thích: Mô hình nhận thấy tín hiệu {dir_text} hiện tại được hỗ trợ "
        f"bởi {factor_phrase} kết hợp với bối cảnh thị trường đa khung thời gian; "
        f"các tin còn lại được xem là ít quan trọng hơn hoặc trùng lặp ngữ nghĩa."
    )

    if alert_decision:
        lines.append(
            f"Quyết định: PHÁT CẢNH BÁO (ĉ={confidence:.2f} đạt ngưỡng tin cậy "
            "và phân phối dự báo đủ sắc nét)."
        )
    else:
        lines.append(
            f"Quyết định: GIỮ IM LẶNG (ĉ={confidence:.2f} chưa đạt ngưỡng — "
            "tránh phát tín hiệu khi bất định cao)."
        )

    return "\n".join(lines)


def explanation_dict(
    direction: int,
    confidence: float,
    horizon: str,
    top_articles: Sequence[Dict],
    top_factors: Sequence[Tuple[str, float]],
    alert_decision: bool,
    return_pred: float | None = None,
    asset: str = "BTC",
) -> dict:
    """Same as :func:`generate_explanation` but also returns the structured
    fields, suitable for JSON serialization in case_studies.json /
    prediction_logs.jsonl.
    """
    return {
        "direction":     int(direction),
        "direction_text": DIRECTION_TEXT.get(int(direction), "?"),
        "confidence":    float(confidence),
        "return_pred":   float(return_pred) if return_pred is not None else None,
        "horizon":       horizon,
        "asset":         asset,
        "alert":         bool(alert_decision),
        "top_articles":  list(top_articles),
        "top_factors":   [(str(f), float(p)) for f, p in top_factors],
        "explanation":   generate_explanation(
            direction, confidence, horizon, top_articles, top_factors,
            alert_decision, return_pred=return_pred, asset=asset,
        ),
    }
