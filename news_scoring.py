# news_scoring.py
from __future__ import annotations

import re
import html
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse, parse_qs

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------
# Utilities
# -----------------------------
KST = timezone(timedelta(hours=9))
LEVEL_ORDER = {"HIGH_ALERT": 0, "WATCHLIST": 1, "FYI": 2, "NOISE": 3}


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_html_tags(s: str) -> str:
    # Naver API title/desc may contain <b>...</b>
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(s)


def normalize_text(s: str, lowercase: bool = True, collapse_whitespace: bool = True) -> str:
    s = strip_html_tags(s)
    if lowercase:
        s = s.lower()
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_pubdate(pub_date: str) -> Optional[datetime]:
    """
    Naver pubDate example: 'Mon, 15 Feb 2026 08:12:00 +0900'
    """
    if not pub_date:
        return None
    try:
        # RFC822-like
        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
        return dt.astimezone(KST)
    except Exception:
        return None


def hours_from_now(dt: datetime, now: datetime) -> float:
    return (now - dt).total_seconds() / 3600.0


def days_from_now(dt: datetime, now: datetime) -> float:
    return (now.date() - dt.date()).days


def extract_domain(url: str) -> str:
    try:
        if not url:
            return ""
        u = urlparse(url)
        host = (u.netloc or "").lower()
        # remove common prefixes
        host = re.sub(r"^www\.", "", host)
        return host
    except Exception:
        return ""


def count_tracking_params(url: str) -> int:
    try:
        if not url:
            return 0
        q = urlparse(url).query
        if not q:
            return 0
        params = parse_qs(q)
        # heuristic: typical tracking keys
        tracking_keys = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
        return sum(1 for k in params.keys() if k.lower() in tracking_keys) + max(0, len(params) - 6)
    except Exception:
        return 0


def normalize_title_for_similarity(title: str, cfg: dict) -> str:
    t = normalize_text(title, lowercase=cfg.get("lowercase", True), collapse_whitespace=True)
    if cfg.get("remove_brackets", True):
        t = re.sub(r"[\[\]\(\)\{\}]", " ", t)
    if cfg.get("remove_press_suffix_patterns", True):
        # Remove common suffix patterns like "… | 연합뉴스" or "…-조선일보"
        t = re.sub(r"\s*[\|\-]\s*[가-힣A-Za-z0-9\.\s]{2,30}$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union else 0.0


@dataclass
class CompatScoredArticle:
    item: Dict[str, Any]
    score: float


def _extract_list(d: dict, path: list[str], default: list[str] | None = None) -> list[str]:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return list(default or [])
        cur = cur.get(key)
    if isinstance(cur, list):
        return [str(x) for x in cur]
    return list(default or [])


def _extract_group_terms(keywords: dict, prefix: str) -> list[str]:
    groups = (keywords or {}).get("groups", {}) or {}
    if not isinstance(groups, dict):
        return []
    pfx = str(prefix or "").upper()
    terms: list[str] = []
    for group_key, node in groups.items():
        gk = str(group_key).upper()
        if not gk.startswith(pfx):
            continue
        group_terms = node.get("terms", []) if isinstance(node, dict) else []
        if not isinstance(group_terms, list):
            continue
        terms.extend(str(t).strip() for t in group_terms if str(t).strip())
    return sorted(set(terms))


def _merge_terms(*term_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for terms in term_lists:
        merged.extend(str(t).strip() for t in (terms or []) if str(t).strip())
    return sorted(set(merged))


def _contains_any(text: str, terms: list[str]) -> bool:
    t = (text or "").lower()
    return any(str(term).lower() in t for term in terms)


def _count_hits(text: str, terms: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for term in terms if str(term).lower() in t)


def _is_els_group(group_key: str) -> bool:
    g = str(group_key).upper()
    return "RISK_ELS" in g or g.startswith("RISK.ELS")


def _assign_level_compat(score: float, thresholds: dict, forced: bool = False) -> str:
    if forced:
        return "HIGH_ALERT"
    if score >= float(thresholds.get("high_alert", 70)):
        return "HIGH_ALERT"
    if score >= float(thresholds.get("watchlist", 50)):
        return "WATCHLIST"
    if score >= float(thresholds.get("fyi", 35)):
        return "FYI"
    return "NOISE"


def _derive_topic(matched_groups: list[str], full_text: str) -> str:
    groups = {str(g).upper() for g in (matched_groups or [])}
    if any(_is_els_group(g) for g in groups):
        return "els"
    if any(g.startswith("LENDING") for g in groups):
        return "lending"
    if any("POLICY" in g for g in groups):
        return "policy_reg"
    if any(g.startswith("COMPETITOR") for g in groups):
        return "competitor"
    if any(g.startswith("PARTNER") for g in groups):
        return "partnership"
    if any("PRODUCT" in g for g in groups):
        return "product"
    if any(g.startswith("GROUP") for g in groups):
        return "company_group"
    if any("BRAND" in g or g.startswith("ENTITY.") for g in groups):
        return "company_group"
    text = (full_text or "").lower()
    if any(t in text for t in ["대출", "주담대", "신용대출", "가계대출", "연체"]):
        return "lending"
    return "other"

def score_article(article: dict, config: dict, keywords: dict, now_kst: Optional[datetime] = None) -> CompatScoredArticle:
    now_kst = now_kst or datetime.now(tz=KST)
    scoring = (config or {}).get("scoring", {}) or {}

    title = str(article.get("title", ""))
    desc = str(article.get("summary", article.get("description", "")))
    full = f"{title} {desc}"

    entities = (keywords or {}).get("entities", {}) or {}
    risk = (keywords or {}).get("risk", {}) or {}
    brand_terms = _merge_terms(
        _extract_group_terms(keywords, "ENTITY."),
        _extract_list({"x": entities}, ["x", "brand"], []),
    )
    els_terms = _merge_terms(
        _extract_group_terms(keywords, "RISK.ELS_ASSET"),
        _extract_list({"x": entities}, ["x", "els_hindex"], []),
    )
    product_terms = _merge_terms(
        _extract_group_terms(keywords, "PRODUCT."),
        _extract_list({"x": entities}, ["x", "product"], []),
    )
    competitor_terms = _merge_terms(
        _extract_group_terms(keywords, "COMPETITOR."),
        _extract_list({"x": entities}, ["x", "competitor"], []),
    )
    partnership_terms = _extract_group_terms(keywords, "PARTNERSHIP.")
    sanction_terms = _merge_terms(
        _extract_group_terms(keywords, "RISK.ELS_SANCTION"),
        _extract_group_terms(keywords, "RISK.GENERAL_ENFORCEMENT"),
        _extract_group_terms(keywords, "POLICY.REGULATOR"),
        _extract_list({"x": risk}, ["x", "sanction"], []),
    )
    dispute_terms = _merge_terms(
        _extract_group_terms(keywords, "RISK.ELS_DISPUTE"),
        _extract_list({"x": risk}, ["x", "dispute"], []),
    )
    loss_terms = _merge_terms(
        _extract_group_terms(keywords, "RISK.ELS_LOSS"),
        _extract_list({"x": risk}, ["x", "loss"], []),
    )
    scale_terms = _merge_terms(
        _extract_group_terms(keywords, "RISK.ESCALATION_SIGNAL"),
        _extract_list({"x": risk}, ["x", "scale"], []),
    )

    base = float(scoring.get("base_score", 10))
    title_weight = float(scoring.get("title_weight", 1.5))
    desc_weight = float(scoring.get("desc_weight", 1.0))
    focus_weights = (scoring.get("focus_weights", {}) or {})
    w_company = float(focus_weights.get("company_group", 1.0))
    w_policy = float(focus_weights.get("policy_reg", 1.0))
    w_els = float(focus_weights.get("els_lending", 1.0))

    # Rebalanced category weights: lower ELS dominance, keep brand/policy competitive.
    group_weights = {
        "BRAND_CORE": 18,
        "BRAND_RISK": 20,
        "GROUP_COMPLIANCE": 17,
        "PRODUCT_VA": 14,
        "PRODUCT_VA_POLICY": 14,
        "PRODUCT_VA_CORE": 14,
        "RISK_ELS_CORE": 14,
        "RISK_ELS_LOSS": 16,
        "RISK_ELS_SANCTION": 18,
        "RISK_ELS_DISPUTE": 16,
        "POLICY_GENERAL_INS": 15,
        "POLICY_CONSUMER": 16,
        "POLICY_WINWIN_FINANCE": 15,
        "POLICY_ELS_CONSUMER": 16,
        "PRODUCT_CPI": 16,
        "PRODUCT_CPI_POLICY": 15,
        "PRODUCT_CPI_CORE": 16,
        "LENDING_CREDIT_INSURANCE": 14,
        "COMPETITOR_CORE": 12,
        "COMPETITOR.CORE": 12,
        "COMPETITOR_RISK": 13,
        "PARTNER_TOSS": 16,
        "PARTNER_FINDA_LOAN": 16,
        "PARTNER_MIRAE_CAPITAL": 15,
        "PARTNER_WOORI_CARD": 15,
    }
    def weighted_group_score(group_key: str) -> float:
        base_w = float(group_weights.get(group_key, 6))
        g = str(group_key).upper()
        if g.startswith("BRAND"):
            return base_w * w_company
        if g.startswith("PARTNER"):
            return base_w * w_company * 0.5
        if g.startswith("COMPETITOR"):
            return base_w * w_company
        if g.startswith("POLICY"):
            return base_w * w_policy
        if "RISK_ELS" in g or g.startswith("RISK"):
            return base_w * w_els
        return base_w

    matched_groups = set(article.get("matched_groups", []))
    if _contains_any(full, competitor_terms):
        matched_groups.add("COMPETITOR.CORE")
    if _contains_any(full, partnership_terms):
        matched_groups.add("PARTNERSHIP.CORE")
    has_els_anchor = _contains_any(full, els_terms)
    # Guardrail: avoid false ELS labeling/scoring when only broad risk tokens hit.
    if not has_els_anchor:
        matched_groups = {g for g in matched_groups if not _is_els_group(g)}
    article["matched_groups"] = sorted(matched_groups)

    group_score = sum(weighted_group_score(g) for g in article.get("matched_groups", []))
    group_score = min(group_score, 28.0)
    mg_upper = {str(g).upper() for g in article.get("matched_groups", [])}
    focus_adjustment = 0.0
    if any(g.startswith("BRAND") for g in mg_upper):
        focus_adjustment += 8.0 * (w_company - 1.0)
    if any(g.startswith("PARTNER") for g in mg_upper):
        focus_adjustment += 4.0 * (w_company - 1.0)
    if any(g.startswith("POLICY") for g in mg_upper):
        focus_adjustment += 6.0 * (w_policy - 1.0)
    if any("RISK_ELS" in g or g.startswith("RISK") for g in mg_upper):
        focus_adjustment += 8.0 * (w_els - 1.0)

    risk_score = 0.0
    if _contains_any(full, els_terms) and _contains_any(full, loss_terms):
        risk_score += 12 * w_els
    if _contains_any(full, els_terms) and _contains_any(full, sanction_terms):
        risk_score += 14 * w_els
    if _contains_any(full, brand_terms) and _contains_any(full, sanction_terms):
        risk_score += 16 * ((w_company + w_policy) / 2.0)
    if _contains_any(full, els_terms) and _contains_any(full, dispute_terms):
        risk_score += 10 * w_els

    context_terms = sorted(
        set(brand_terms + competitor_terms + els_terms + product_terms + partnership_terms + sanction_terms + dispute_terms + loss_terms)
    )
    context_score = _count_hits(title, context_terms) * title_weight + _count_hits(desc, context_terms) * desc_weight

    recency_cfg = scoring.get("recency", {}) or {}
    dt = article.get("published_at_dt")
    recency_score = float(recency_cfg.get("older", 0))
    if isinstance(dt, datetime):
        hours = max((now_kst - dt).total_seconds() / 3600.0, 0.0)
        if hours <= 6:
            recency_score = float(recency_cfg.get("hours_0_6", 8))
        elif hours <= 24:
            recency_score = float(recency_cfg.get("hours_6_24", 6))
        elif hours <= 48:
            recency_score = float(recency_cfg.get("days_1_2", 3))
        elif hours <= 72:
            recency_score = float(recency_cfg.get("days_2_3", 1))

    noise_kw = dict(scoring.get("noise_keywords", {}) or {})
    noise_kw.setdefault("strong", [])
    noise_kw.setdefault("medium", [])
    noise_kw.setdefault("weak", [])
    # Reflect keywords.yaml groups in noise penalties as well.
    noise_kw["strong"] = _merge_terms(noise_kw["strong"], _extract_group_terms(keywords, "NOISE.STRONG"))
    noise_kw["medium"] = _merge_terms(noise_kw["medium"], _extract_group_terms(keywords, "NOISE.MEDIUM"))
    noise_kw["weak"] = _merge_terms(noise_kw["weak"], _extract_group_terms(keywords, "NOISE.WEAK"))
    noise_penalty_cfg = scoring.get("noise_penalty", {}) or {}
    sanction_hit = _contains_any(full, sanction_terms)
    noise_penalty = 0.0
    for level in ("strong", "medium", "weak"):
        kws = [str(x) for x in (noise_kw.get(level, []) or [])]
        if _contains_any(full, kws):
            p = float(noise_penalty_cfg.get(level, 0))
            noise_penalty += (p / 2.0) if sanction_hit else p

    link = str(article.get("originallink") or article.get("link") or "")
    domain = extract_domain(link)
    source_cfg = (scoring.get("source_scoring", {}) or {})
    scores = (source_cfg.get("scores", {}) or {})
    tier1_score = float(scores.get("tier_1", 10))
    tier2_score = float(scores.get("tier_2", 4))
    unknown_score = float(scores.get("unknown", 1))
    suspicious_score = float(scores.get("suspicious", -3))
    tier1_patterns = source_cfg.get(
        "tier_1_contains_any",
        ["yna", "yonhap", "mk.co.kr", "hankyung", "sedaily", "donga", "chosun", "joongang", "hani", "khan", "fnnews", "mt.co.kr", "newsis", "etnews"],
    )
    tier2_patterns = source_cfg.get("tier_2_contains_any", ["news", "press", "biz", "economy"])

    source = unknown_score
    if not domain:
        source = suspicious_score
    elif any(str(p) in domain for p in tier1_patterns):
        source = tier1_score
    elif any(str(p) in domain for p in tier2_patterns):
        source = tier2_score

    forced = False
    if _contains_any(full, els_terms) and _contains_any(full, sanction_terms):
        forced = True
    if _contains_any(full, brand_terms) and _contains_any(full, sanction_terms):
        forced = True
    if _contains_any(full, els_terms) and _contains_any(full, dispute_terms) and _contains_any(full, scale_terms):
        forced = True

    forced_add = float(((scoring.get("forced_high_alert", {}) or {}).get("add_score", 30))) if forced else 0.0
    final_score = base + group_score + risk_score + context_score + recency_score + source - noise_penalty + forced_add + focus_adjustment

    thresholds = scoring.get("thresholds", {}) or {}
    level = _assign_level_compat(final_score, thresholds, forced=forced)

    article["press_domain"] = domain
    article["source_score"] = source
    article["score"] = int(round(final_score))
    article["priority_score"] = int(round(final_score))
    article["level"] = level
    article["summary_note"] = article.get("summary_note", "")
    article["matched_groups"] = sorted(set(article.get("matched_groups", [])))
    article["topic"] = _derive_topic(article["matched_groups"], full)
    article["priority_terms"] = ", ".join(article["matched_groups"][:3]) if article["matched_groups"] else "general"
    article["forced_high_alert"] = forced

    return CompatScoredArticle(item=article, score=final_score)


def dedupe_and_cluster(scored_articles: List[CompatScoredArticle], config: dict) -> List[CompatScoredArticle]:
    scoring = (config or {}).get("scoring", {}) or {}
    dup_cfg = scoring.get("duplicate_penalty", {}) or {}
    same_link_penalty = float(dup_cfg.get("same_originallink", 100))
    similar_penalty = float(dup_cfg.get("similar", 15))
    thresholds = scoring.get("thresholds", {}) or {}

    seen_links: set[str] = set()
    clusters: List[CompatScoredArticle] = []
    cluster_seq = 1

    def jaccard(a: str, b: str) -> float:
        sa = set(re.sub(r"[^0-9a-zA-Z가-힣\\s]", " ", (a or "").lower()).split())
        sb = set(re.sub(r"[^0-9a-zA-Z가-힣\\s]", " ", (b or "").lower()).split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    for row in sorted(scored_articles, key=lambda x: x.score, reverse=True):
        item = row.item
        link = str(item.get("originallink") or item.get("link") or "").strip()
        if link and link in seen_links:
            item["score"] = int(round(item.get("score", 0) - same_link_penalty))
            if item.get("level") != "HIGH_ALERT":
                item["level"] = _assign_level_compat(item["score"], thresholds, forced=False)
            continue
        if link:
            seen_links.add(link)

        assigned = False
        for existing in clusters:
            if jaccard(item.get("title", ""), existing.item.get("title", "")) >= 0.72:
                item["cluster_id"] = existing.item.get("cluster_id")
                item["score"] = int(round(item.get("score", 0) - similar_penalty))
                if item.get("level") != "HIGH_ALERT":
                    item["level"] = _assign_level_compat(item["score"], thresholds, forced=False)
                assigned = True
                break

        if assigned:
            continue

        item["cluster_id"] = f"C{cluster_seq:04d}"
        cluster_seq += 1
        clusters.append(CompatScoredArticle(item=item, score=float(item.get("score", row.score))))

    clusters.sort(
        key=lambda r: (
            LEVEL_ORDER.get(r.item.get("level", "NOISE"), 9),
            -float(r.item.get("score", 0)),
            -(r.item.get("published_at_dt").timestamp() if isinstance(r.item.get("published_at_dt"), datetime) else 0),
        )
    )
    return clusters




