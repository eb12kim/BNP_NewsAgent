import copy
import csv
import html
import json
import mimetypes
import os
import re
import smtplib
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import schedule
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from news_scoring import LEVEL_ORDER, dedupe_and_cluster, score_article

# Set via environment variables or Streamlit secrets.
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "").strip()
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "").strip()

API_URL = "https://openapi.naver.com/v1/search/news.json"
TOP10_FILE = "news_report.csv"
CANDIDATES_FILE = "news_candidates_100.csv"
DRAFT_EDITS_FILE = "news_draft_edits.json"
DISPLAY_PER_QUERY = 100
KST = ZoneInfo("Asia/Seoul")
SETTINGS_FILE = "settings.json"
BODY_VALIDATION_CACHE_FILE = "body_validation_cache.json"
NEWS_MONITORING_CONFIG_FILE = "config/news_monitoring.yaml"
KEYWORDS_CONFIG_FILE = "config/keywords.yaml"
OUTPUT_DIR = "."

# Set to True for immediate verification run. Set to False to use the scheduler.
RUN_ONCE_FOR_TEST = True
SCHEDULE_TIME = "09:00"

# For this run, collect from the latest 7 days (excluding today).
LOOKBACK_DAYS = 7

# Candidate target: at least 50 and up to 100.
CANDIDATE_COUNT = 100
VALIDATION_TOP_K = 120
BODY_CACHE_TTL_SECONDS = 60 * 60 * 24 * 2
QUICK_MIN_TEXT_LEN = 120
VALIDATION_FETCH_TIMEOUT = 5
FOCUS_WEIGHTS = {
    "company_group": 1.0,
    "policy_reg": 1.0,
    "els_lending": 1.0,
}
TOPIC_QUOTA_RATIO = {
    "company_group": 0.35,
    "policy_reg": 0.25,
    "lending": 0.2,
    "els": 0.1,
    "product": 0.1,
}

# Optional: fill for auto email delivery.
SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASSWORD = ""
EMAIL_FROM = ""
EMAIL_TO = []  # e.g. ["all@company.com"]


SENSITIVE_TERMS = ["m&a", "인수합병", "인수", "합병", "매각", "지분매각", "경영권", "인수전", "적대적", "매물", "lbo"]
SPORTS_NOISE_TERMS = [
    "축구",
    "야구",
    "농구",
    "배구",
    "골프",
    "e스포츠",
    "esports",
    "구단",
    "선수",
    "감독",
    "코치",
    "경기",
    "리그",
    "K리그",
    "KBO",
    "KBL",
    "WKBL",
    "챔피언스리그",
    "유니폼",
    "후원",
    "스폰서",
    "스폰서십",
    "배구단",
    "V리그",
    "브이리그",
    "세트스코어",
    "득점",
    "라운드",
    "챔프전",
    "플레이오프",
]
FINANCE_KEEP_TERMS = [
    "보험",
    "생명보험",
    "신용보험",
    "단체신용보험",
    "대출",
    "대출안심",
    "금감원",
    "금융위원회",
    "금융감독원",
    "정책",
    "제재",
    "민원",
    "소비자보호",
]
STRONG_FINANCE_TERMS = [
    "신용보험",
    "단체신용보험",
    "신용생명보험",
    "대출안심보험",
    "대출안심보장보험",
    "금융소비자보호법",
    "보험업감독규정",
    "검사",
    "제재",
    "과징금",
]


def _keyword_group_terms(group_key: str, fallback: list[str]) -> list[str]:
    groups = KEYWORDS_CONFIG.get("groups", {}) if isinstance(KEYWORDS_CONFIG, dict) else {}
    node = groups.get(group_key, {}) if isinstance(groups, dict) else {}
    terms = node.get("terms", []) if isinstance(node, dict) else []
    cleaned = [str(t).strip() for t in terms if str(t).strip()]
    return cleaned if cleaned else list(fallback)

MEDIA_PRIORITY = {
    "yna.co.kr": 100,
    "newsis.com": 95,
    "mk.co.kr": 95,
    "hankyung.com": 95,
    "sedaily.com": 90,
    "chosun.com": 90,
    "joongang.co.kr": 90,
    "donga.com": 90,
    "hani.co.kr": 88,
    "khan.co.kr": 88,
    "edaily.co.kr": 88,
    "fnnews.com": 88,
    "news.naver.com": 85,
    "biz.chosun.com": 85,
    "mt.co.kr": 85,
    "etnews.com": 84,
    "seoul.co.kr": 82,
}

BASE_TOPIC_QUOTA_RATIO = copy.deepcopy(TOPIC_QUOTA_RATIO)
BASE_FOCUS_WEIGHTS = copy.deepcopy(FOCUS_WEIGHTS)

PRESET_SETTINGS = {
    "balanced": {
        "lookback_days": 3,
        "candidate_count": 100,
        "focus_weights": {"company_group": 1.0, "policy_reg": 1.0, "els_lending": 1.0},
    },
    "company_first": {
        "lookback_days": 3,
        "candidate_count": 100,
        "focus_weights": {"company_group": 1.4, "policy_reg": 1.0, "els_lending": 0.9},
    },
    "policy_first": {
        "lookback_days": 3,
        "candidate_count": 100,
        "focus_weights": {"company_group": 1.0, "policy_reg": 1.4, "els_lending": 1.0},
    },
    "els_lending_first": {
        "lookback_days": 3,
        "candidate_count": 100,
        "focus_weights": {"company_group": 0.9, "policy_reg": 1.0, "els_lending": 1.5},
    },
}


def _default_news_monitoring_config() -> dict:
    return {
        "days_back": 3,
        "query_groups": [
            {"key": "BRAND_CORE", "label": "Brand - Core", "query": '"BNP파리바카디프생명" | "카디프생명"'},
            {
                "key": "BRAND_RISK",
                "label": "Brand - Risk",
                "query": '("BNP파리바카디프생명" | "카디프생명") (민원 | 제재 | 과징금 | 소송 | 분쟁 | 피해 | 불완전판매 | 금감원 | 소비자보호)',
            },
            {
                "key": "PRODUCT_VA",
                "label": "Product - Variable (ELS/ETF)",
                "query": '("BNP파리바카디프생명" | "카디프생명") ("ELS변액보험" | "ETF변액보험" | "변액보험" | "ELS연계" | "ETF연계")',
            },
            {
                "key": "RISK_ELS_CORE",
                "label": "Risk - ELS/H Index Loss",
                "query": '("ELS" | "홍콩H지수" | "H지수") (원금손실 | 손실확정 | 대규모손실 | 녹인 | 배리어 | 손해 | 피해)',
            },
            {
                "key": "RISK_ELS_SANCTION",
                "label": "Risk - ELS Sanction/Inspection",
                "query": '("ELS" | "홍콩H지수" | "H지수") (금감원 | 금융위원회 | 검사 | 제재 | 징계 | 과징금 | 기관경고 | 시정명령)',
            },
            {
                "key": "POLICY_ELS_CONSUMER",
                "label": "Policy - Consumer Protection (ELS)",
                "query": '("ELS" | "홍콩H지수" | "H지수") (금융소비자보호법 | 불완전판매 | 내부통제 | 적합성 | 적정성 | 설명의무 | 판매책임)',
            },
            {
                "key": "PRODUCT_CPI",
                "label": "Product - CPI",
                "query": '("신용생명보험" | "단체신용보험" | "CPI" | "대출안심보장보험") (보험사 | 금융사 | 카드사 | 저축은행)',
            },
        ],
        "scoring": {
            "base_score": 10,
            "title_weight": 1.5,
            "desc_weight": 1.0,
            "recency": {"hours_0_6": 8, "hours_6_24": 6, "days_1_2": 3, "days_2_3": 1, "older": 0},
            "noise_keywords": {
                "strong": ["리포트", "목표주가", "차트", "급등", "급락", "종목", "매수", "매도"],
                "medium": ["증시", "코스피", "코스닥", "나스닥", "다우"],
                "weak": ["주가"],
            },
            "noise_penalty": {"strong": 12, "medium": 8, "weak": 6},
            "duplicate_penalty": {"same_originallink": 100, "similar": 15},
            "forced_high_alert": {"add_score": 30},
            "thresholds": {"high_alert": 70, "watchlist": 50, "fyi": 35},
        },
    }


def _default_keywords() -> dict:
    return {
        "entities": {
            "brand": ["BNP파리바카디프생명", "카디프생명", "BNP Paribas Cardif"],
            "els_hindex": ["ELS", "홍콩H지수", "H지수"],
            "product": ["ELS변액보험", "ETF변액보험", "변액보험", "ELS연계", "ETF연계", "신용생명보험", "단체신용보험", "CPI", "대출안심보장보험"],
        },
        "risk": {
            "sanction": ["금감원", "금융감독원", "금융위원회", "검사", "제재", "징계", "과징금"],
            "dispute": ["배상", "분쟁조정", "민원", "소송"],
            "loss": ["원금손실", "손실확정", "대규모손실", "녹인", "배리어", "손해", "피해"],
            "scale": ["대규모", "집단", "확정", "비율"],
        },
    }


def _load_yaml_with_fallback(path: str, fallback: dict) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists() or yaml is None:
        return copy.deepcopy(fallback)
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    return data if isinstance(data, dict) else copy.deepcopy(fallback)


NEWS_MONITORING_CONFIG = _load_yaml_with_fallback(NEWS_MONITORING_CONFIG_FILE, _default_news_monitoring_config())
KEYWORDS_CONFIG = _load_yaml_with_fallback(KEYWORDS_CONFIG_FILE, _default_keywords())


def _effective_days_back_from_config() -> int:
    raw = NEWS_MONITORING_CONFIG.get("days_back")
    if raw is None:
        runtime = NEWS_MONITORING_CONFIG.get("runtime", {}) if isinstance(NEWS_MONITORING_CONFIG.get("runtime"), dict) else {}
        raw = runtime.get("default_days_back", LOOKBACK_DAYS)
    try:
        return max(1, int(raw))
    except Exception:
        return 3


def resolve_output_dir(path_value: str | None) -> Path:
    raw = str(path_value or ".").strip() or "."
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        path = Path.cwd()
    return path


def resolve_output_file_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = resolve_output_dir(OUTPUT_DIR) / path
    return path


def get_default_settings() -> dict:
    return {
        "preset": "balanced",
        "lookback_days": _effective_days_back_from_config(),
        "candidate_count": CANDIDATE_COUNT,
        "focus_weights": copy.deepcopy(BASE_FOCUS_WEIGHTS),
        "topic_quota_ratio": copy.deepcopy(BASE_TOPIC_QUOTA_RATIO),
        "output_dir": str(resolve_output_dir(OUTPUT_DIR)),
    }


def apply_runtime_settings(settings: dict) -> None:
    global LOOKBACK_DAYS, CANDIDATE_COUNT, TOPIC_QUOTA_RATIO, FOCUS_WEIGHTS, OUTPUT_DIR

    LOOKBACK_DAYS = max(1, int(settings.get("lookback_days", LOOKBACK_DAYS)))
    CANDIDATE_COUNT = int(settings.get("candidate_count", CANDIDATE_COUNT))
    OUTPUT_DIR = str(resolve_output_dir(settings.get("output_dir", OUTPUT_DIR)))
    NEWS_MONITORING_CONFIG["days_back"] = LOOKBACK_DAYS
    runtime_cfg = NEWS_MONITORING_CONFIG.get("runtime")
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
        NEWS_MONITORING_CONFIG["runtime"] = runtime_cfg
    runtime_cfg["default_days_back"] = LOOKBACK_DAYS
    runtime_cfg["max_days_back"] = max(3, LOOKBACK_DAYS)

    merged_weights = copy.deepcopy(BASE_FOCUS_WEIGHTS)
    merged_weights.update(settings.get("focus_weights", {}))
    FOCUS_WEIGHTS = merged_weights

    quota = settings.get("topic_quota_ratio", {})
    if isinstance(quota, dict) and quota:
        merged_quota = copy.deepcopy(BASE_TOPIC_QUOTA_RATIO)
        for k, v in quota.items():
            try:
                merged_quota[k] = float(v)
            except Exception:
                continue
        TOPIC_QUOTA_RATIO = merged_quota


def load_settings(file_path: str = SETTINGS_FILE) -> dict:
    default_settings = get_default_settings()
    path = Path(file_path)
    if not path.exists():
        apply_runtime_settings(default_settings)
        return default_settings

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    settings = copy.deepcopy(default_settings)
    settings.update({k: v for k, v in data.items() if k in settings})
    if isinstance(data.get("focus_weights"), dict):
        settings["focus_weights"].update(data["focus_weights"])
    if isinstance(data.get("topic_quota_ratio"), dict):
        settings["topic_quota_ratio"].update(data["topic_quota_ratio"])

    apply_runtime_settings(settings)
    return settings


def save_settings(settings: dict, file_path: str = SETTINGS_FILE) -> None:
    path = Path(file_path)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def build_settings_with_preset(
    preset: str,
    lookback_days: int,
    candidate_count: int,
    focus_weights: dict | None = None,
    output_dir: str | None = None,
) -> dict:
    settings = get_default_settings()
    base = PRESET_SETTINGS.get(preset, PRESET_SETTINGS["balanced"])
    settings["preset"] = preset if preset in PRESET_SETTINGS else "balanced"
    settings["lookback_days"] = int(base.get("lookback_days", settings["lookback_days"]))
    settings["candidate_count"] = int(base.get("candidate_count", settings["candidate_count"]))
    settings["focus_weights"].update(base.get("focus_weights", {}))

    settings["lookback_days"] = int(lookback_days)
    settings["candidate_count"] = int(candidate_count)
    if isinstance(focus_weights, dict):
        settings["focus_weights"].update(focus_weights)
    if output_dir is not None:
        settings["output_dir"] = str(resolve_output_dir(output_dir))
    return settings


def clean_html_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def parse_pub_date_to_kst(pub_date: str) -> datetime | None:
    if not pub_date:
        return None
    try:
        dt = parsedate_to_datetime(pub_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    except Exception:
        return None


def get_target_dates(today_kst: datetime, lookback_days: int | None = None) -> set:
    today = today_kst.date()
    days = max(1, lookback_days if lookback_days is not None else LOOKBACK_DAYS)
    return {today - timedelta(days=i) for i in range(1, days + 1)}


def extract_domain(link: str) -> str:
    domain = urlparse(link).netloc.lower() if link else ""
    return domain[4:] if domain.startswith("www.") else domain


def get_media_score(domain: str) -> int:
    if domain in MEDIA_PRIORITY:
        return MEDIA_PRIORITY[domain]
    for known, score in MEDIA_PRIORITY.items():
        if domain.endswith(known):
            return score
    return 20


def fetch_news_by_keyword(keyword: str, group_key: str) -> list[dict]:
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": keyword, "display": DISPLAY_PER_QUERY, "start": 1, "sort": "date"}
    response = requests.get(API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()

    results = []
    for item in response.json().get("items", []):
        link = item.get("originallink") or item.get("link", "")
        published_dt = parse_pub_date_to_kst(item.get("pubDate", ""))
        results.append(
            {
                "title": clean_html_text(item.get("title", "")),
                "link": link,
                "summary": clean_html_text(item.get("description", "")),
                "published_at_dt": published_dt,
                "published_at": published_dt.strftime("%Y-%m-%d %H:%M:%S") if published_dt else "",
                "matched_keywords": {keyword},
                "matched_groups": [group_key] if group_key else [],
                "media_domain": extract_domain(link),
                "media_score": get_media_score(extract_domain(link)),
                "press_domain": extract_domain(link),
            }
        )
    return results


def merge_and_deduplicate(news_batches: list[list[dict]]) -> list[dict]:
    merged = {}
    for batch in news_batches:
        for item in batch:
            key = item["link"] or item["title"]
            if not key:
                continue
            if key in merged:
                merged[key]["matched_keywords"].update(item["matched_keywords"])
                merged[key]["matched_groups"] = sorted(
                    set(merged[key].get("matched_groups", [])) | set(item.get("matched_groups", []))
                )
            else:
                merged[key] = item
    return list(merged.values())


def filter_by_days_back(news_items: list[dict], now_kst: datetime, days_back: int) -> list[dict]:
    cutoff = now_kst - timedelta(days=max(1, int(days_back)))
    return [item for item in news_items if item.get("published_at_dt") and item["published_at_dt"] >= cutoff]


def contains_sensitive_topic(item: dict) -> bool:
    text_blob = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    return any(term in text_blob for term in SENSITIVE_TERMS)


def is_sports_noise_article(item: dict) -> bool:
    """
    Exclude sports sponsorship/team articles that match partner keywords
    but are unrelated to insurance/financial monitoring.
    """
    text_blob = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    sports_terms = _keyword_group_terms("NOISE.SPORTS", SPORTS_NOISE_TERMS)
    finance_keep_terms = _keyword_group_terms("FILTER.FINANCE_KEEP", FINANCE_KEEP_TERMS)
    strong_finance_terms = _keyword_group_terms("FILTER.STRONG_FINANCE_KEEP", STRONG_FINANCE_TERMS)
    sports_hits = sum(1 for term in sports_terms if term.lower() in text_blob)
    if sports_hits == 0:
        return False

    # For partner queries, sports context is a hard exclude by default to avoid sponsor-team noise.
    matched_groups = {str(g).upper() for g in item.get("matched_groups", [])}
    if any(g.startswith("PARTNER_") for g in matched_groups):
        if any(term.lower() in text_blob for term in strong_finance_terms):
            return False
        return True

    # Non-partner queries: keep only if finance context is clearly present.
    finance_hits = sum(1 for term in finance_keep_terms if term.lower() in text_blob)
    if finance_hits >= 2:
        return False
    return True


def apply_topic_quotas(news_items: list[dict], target_count: int) -> list[dict]:
    quotas = {topic: max(1, int(target_count * ratio)) for topic, ratio in TOPIC_QUOTA_RATIO.items()}
    selected_keys = set()
    selected = []

    for topic in ["company_group", "policy_reg", "lending", "els", "product"]:
        count = 0
        for item in news_items:
            if item.get("topic") != topic:
                continue
            key = item.get("link") or item.get("title")
            if not key or key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            count += 1
            if count >= quotas[topic]:
                break

    for item in news_items:
        if len(selected) >= target_count:
            break
        key = item.get("link") or item.get("title")
        if not key or key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)

    return selected


def extract_article_text_from_html(page_html: str) -> str:
    body = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", page_html)
    body = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", body)

    article_blocks = re.findall(
        r'(?is)<(?:article|section|div)[^>]+(?:id|class)=["\'][^"\']*(?:article|news|content|post)[^"\']*["\'][^>]*>(.*?)</(?:article|section|div)>',
        body,
    )
    source_html = " ".join(article_blocks) if article_blocks else body

    paras = [clean_html_text(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", source_html)]
    if not paras:
        paras = [clean_html_text(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)]

    boilerplate_patterns = [
        r"internet explorer",
        r"최신 브라우저",
        r"브라우저.*권장",
        r"현재 internet explorer",
        r"무단전재",
        r"저작권자",
        r"기사제보",
        r"구독",
        r"광고문의",
        r"본문.*무단",
    ]

    filtered = []
    for p in paras:
        if len(p) < 40:
            continue
        lower = p.lower()
        if any(re.search(pattern, lower) for pattern in boilerplate_patterns):
            continue
        filtered.append(p)

    text = "\n\n".join(filtered).strip()
    return text if text else "본문을 추출하지 못했습니다. 원문 링크를 확인해 주세요."


def fetch_article_body(link: str) -> str:
    try:
        response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return extract_article_text_from_html(response.text)
    except Exception:
        return "본문을 가져오지 못했습니다. 원문 링크를 확인해 주세요."


def is_valid_article_body_fast(link: str, cache: dict[str, bool]) -> bool:
    if not link:
        return False
    if link in cache:
        return cache[link]

    try:
        response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=VALIDATION_FETCH_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        html_text = response.text[:200000]
        body = extract_article_text_from_html(html_text)
    except Exception:
        cache[link] = False
        return False

    invalid_markers = [
        "본문을 추출하지 못했습니다",
        "본문을 가져오지 못했습니다",
        "internet explorer",
        "최신 브라우저",
        "브라우저 사용을 권장",
    ]
    lowered = body.lower()
    if any(marker in lowered for marker in invalid_markers):
        cache[link] = False
        return False

    normalized = re.sub(r"\s+", " ", body).strip()
    valid = len(normalized) >= 100
    cache[link] = valid
    return valid


def load_body_validation_cache(file_path: str = BODY_VALIDATION_CACHE_FILE) -> dict:
    path = Path(file_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_body_validation_cache(cache_data: dict, file_path: str = BODY_VALIDATION_CACHE_FILE) -> None:
    path = Path(file_path)
    path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")


def quick_quality_pass(item: dict) -> bool:
    title = item.get("title", "")
    summary = item.get("summary", "")
    combined = re.sub(r"\s+", " ", f"{title} {summary}").strip()
    if len(combined) < QUICK_MIN_TEXT_LEN:
        return False

    bad_patterns = [
        r"internet explorer",
        r"최신 브라우저",
        r"브라우저.*권장",
        r"무단전재",
        r"저작권자",
    ]
    lower = combined.lower()
    return not any(re.search(p, lower) for p in bad_patterns)


def is_valid_article_body_cached(
    link: str,
    memory_cache: dict[str, bool],
    disk_cache: dict,
    now_ts: int,
) -> bool:
    if link in memory_cache:
        return memory_cache[link]

    cached = disk_cache.get(link)
    if isinstance(cached, dict):
        ts = int(cached.get("ts", 0) or 0)
        if now_ts - ts <= BODY_CACHE_TTL_SECONDS:
            valid = bool(cached.get("valid", False))
            memory_cache[link] = valid
            return valid

    valid = is_valid_article_body_fast(link, memory_cache)
    disk_cache[link] = {"valid": valid, "ts": now_ts}
    return valid


def select_candidates_with_fast_validation(ordered_items: list[dict], target_count: int) -> list[dict]:
    # Stage 1: ultra-fast filter using only title/summary.
    quick_passed = [item for item in ordered_items if quick_quality_pass(item)]

    # Stage 2: validate article body only for top-ranked subset.
    memory_cache: dict[str, bool] = {}
    disk_cache = load_body_validation_cache()
    now_ts = int(time.time())

    selected = []
    validated = 0
    for item in quick_passed:
        if len(selected) >= target_count:
            break
        link = item.get("link", "")
        if validated < VALIDATION_TOP_K:
            validated += 1
            if not is_valid_article_body_cached(link, memory_cache, disk_cache, now_ts):
                continue
        selected.append(item)

    save_body_validation_cache(disk_cache)
    return selected


def make_safe_output_path(file_path: str) -> str:
    path = resolve_output_file_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return str(path.resolve())
    except PermissionError:
        stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        return str(fallback.resolve())


def save_draft_edits(draft_items: list[dict], file_path: str = DRAFT_EDITS_FILE) -> str:
    safe_path = make_safe_output_path(file_path)
    rows = []
    for idx, item in enumerate(draft_items, start=1):
        row = {
            "order": idx,
            "candidate_id": int(item.get("candidate_id", 0) or 0),
            "title": str(item.get("title", "") or ""),
            "summary_note": str(item.get("summary_note", "") or ""),
            "body_override": str(item.get("body_override", "") or ""),
            "link": str(item.get("link", "") or ""),
            "published_at": str(item.get("published_at", "") or ""),
            "media_domain": str(item.get("media_domain", "") or ""),
        }
        rows.append(row)
    Path(safe_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return safe_path


def load_draft_edits(file_path: str = DRAFT_EDITS_FILE) -> list[dict]:
    path = resolve_output_file_path(file_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    result = []
    for row in data:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "candidate_id": int(row.get("candidate_id", 0) or 0),
                "title": str(row.get("title", "") or ""),
                "summary_note": str(row.get("summary_note", "") or ""),
                "body_override": str(row.get("body_override", "") or ""),
                "link": str(row.get("link", "") or ""),
                "published_at": str(row.get("published_at", "") or ""),
                "media_domain": str(row.get("media_domain", "") or ""),
            }
        )
    return result


def save_candidates_to_csv(news_items: list[dict], file_path: str) -> str:
    safe_path = make_safe_output_path(file_path)
    fields = [
        "candidate_id",
        "score",
        "level",
        "press_domain",
        "cluster_id",
        "matched_groups",
        "priority_score",
        "media_domain",
        "published_at",
        "title",
        "link",
        "summary",
        "summary_note",
        "topic",
        "why_selected",
        "cluster_size",
        "priority_terms",
        "matched_keywords",
        "publish_yn",
    ]
    with open(safe_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, item in enumerate(news_items, start=1):
            item["candidate_id"] = idx
            writer.writerow(
                {
                    "candidate_id": idx,
                    "score": item.get("score", item.get("priority_score", 0)),
                    "level": item.get("level", "NOISE"),
                    "press_domain": item.get("press_domain", item.get("media_domain", "")),
                    "cluster_id": item.get("cluster_id", ""),
                    "matched_groups": ", ".join(item.get("matched_groups", [])),
                    "priority_score": item.get("priority_score", 0),
                    "media_domain": item.get("media_domain", ""),
                    "published_at": item.get("published_at", ""),
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "summary": item.get("summary", ""),
                    "summary_note": item.get("summary_note", ""),
                    "topic": item.get("topic", ""),
                    "why_selected": item.get("why_selected", ""),
                    "cluster_size": item.get("cluster_size", 1),
                    "priority_terms": item.get("priority_terms", ""),
                    "matched_keywords": ", ".join(sorted(item.get("matched_keywords", set()))),
                    "publish_yn": "",
                }
            )
    return safe_path


def read_selected_from_candidates(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        return []

    selected = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("publish_yn", "").strip().upper() != "Y":
                continue
            try:
                candidate_id = int(row.get("candidate_id", "0"))
            except ValueError:
                candidate_id = 0
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "score": int(row.get("score", row.get("priority_score", "0")) or 0),
                    "level": row.get("level", "NOISE"),
                    "press_domain": row.get("press_domain", row.get("media_domain", "")),
                    "cluster_id": row.get("cluster_id", ""),
                    "matched_groups": [
                        token.strip() for token in row.get("matched_groups", "").split(",") if token.strip()
                    ],
                    "priority_score": int(row.get("priority_score", "0") or 0),
                    "media_domain": row.get("media_domain", ""),
                    "published_at": row.get("published_at", ""),
                    "title": row.get("title", ""),
                    "link": row.get("link", ""),
                    "summary": row.get("summary", ""),
                    "summary_note": row.get("summary_note", ""),
                    "topic": row.get("topic", ""),
                    "why_selected": row.get("why_selected", ""),
                    "priority_terms": row.get("priority_terms", ""),
                    "matched_keywords": {
                        token.strip() for token in row.get("matched_keywords", "").split(",") if token.strip()
                    },
                }
            )

    selected.sort(key=lambda x: (x.get("candidate_id", 0)))
    return selected


def has_publish_mark(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("publish_yn", "").strip().upper() == "Y":
                return True
    return False


def find_candidates_file_for_publish(base_file: str) -> str:
    base_path = resolve_output_file_path(base_file)
    pattern = f"{base_path.stem}*.csv"
    candidates = sorted(base_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if has_publish_mark(str(path)):
            return str(path)
    return str(base_path)


def enforce_selected_rules(news_items: list[dict], max_per_keyword: int = 3) -> list[dict]:
    for item in news_items:
        if item.get("matched_keywords"):
            item["primary_keyword"] = sorted(item["matched_keywords"])[0]
        else:
            item["primary_keyword"] = ""
        item["priority_bucket"] = 0 if item.get("topic") == "company_group" else 1

    ordered = sorted(
        news_items,
        key=lambda x: (
            x.get("priority_bucket", 1),
            -(x.get("priority_score", 0)),
            x.get("candidate_id", 0),
        ),
    )

    keyword_counts = {}
    filtered = []
    for item in ordered:
        pk = item.get("primary_keyword", "")
        if pk:
            if keyword_counts.get(pk, 0) >= max_per_keyword:
                continue
            keyword_counts[pk] = keyword_counts.get(pk, 0) + 1
        filtered.append(item)
    return filtered


def enforce_keyword_media_cap(news_items: list[dict], max_per_pair: int = 2) -> list[dict]:
    def primary_keyword(item: dict) -> str:
        if item.get("primary_keyword"):
            return str(item.get("primary_keyword"))
        keywords = sorted(set(item.get("matched_keywords", set())))
        if not keywords:
            return ""
        return keywords[0]

    def has_comprehensive_marker(title: str) -> bool:
        text = str(title or "")
        return "(종합)" in text or "[종합]" in text or " 종합 " in f" {text} "

    grouped: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    pass_through_indexes: set[int] = set()

    for idx, item in enumerate(news_items):
        keyword = primary_keyword(item)
        media = str(item.get("press_domain") or item.get("media_domain") or "")
        if not keyword or not media:
            pass_through_indexes.add(idx)
            continue
        grouped.setdefault((keyword, media), []).append((idx, item))

    selected_indexes: set[int] = set(pass_through_indexes)
    for rows in grouped.values():
        # Prefer "(종합)" titles first, then keep original ranking order.
        picked = sorted(rows, key=lambda x: (0 if has_comprehensive_marker(x[1].get("title", "")) else 1, x[0]))[:max_per_pair]
        selected_indexes.update(idx for idx, _ in picked)

    return [item for idx, item in enumerate(news_items) if idx in selected_indexes]


def enforce_topic_cap(news_items: list[dict], max_per_topic: int = 20) -> list[dict]:
    counts: dict[str, int] = {}
    selected: list[dict] = []
    for item in news_items:
        topic = str(item.get("topic", "other") or "other")
        if counts.get(topic, 0) >= max_per_topic:
            continue
        counts[topic] = counts.get(topic, 0) + 1
        selected.append(item)
    return selected


def save_top10_csv(news_items: list[dict], file_path: str) -> str:
    safe_path = make_safe_output_path(file_path)
    fields = [
        "candidate_id",
        "score",
        "level",
        "press_domain",
        "cluster_id",
        "matched_groups",
        "priority_score",
        "media_domain",
        "published_at",
        "title",
        "link",
        "summary",
        "summary_note",
        "topic",
        "why_selected",
        "priority_terms",
    ]
    with open(safe_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in news_items:
            row = {key: item.get(key, "") for key in fields}
            if isinstance(row.get("matched_groups"), list):
                row["matched_groups"] = ", ".join(row["matched_groups"])
            writer.writerow(row)
    return safe_path


def pick_selected_news(candidates: list[dict], selected_ids: list[int]) -> list[dict]:
    selected_map = {item.get("candidate_id"): item for item in candidates}
    selected = []
    for cid in selected_ids:
        item = selected_map.get(cid)
        if item:
            selected.append(item)
    return selected


def parse_ids_input(raw: str) -> list[int]:
    ids = []
    for token in re.split(r"[\s,]+", raw.strip()):
        if token.isdigit():
            ids.append(int(token))
    return sorted(set(ids))


def print_candidate_preview(candidates: list[dict], limit: int = 20) -> None:
    print("\n[후보 미리보기]")
    print("ID | 토픽 | 점수 | 제목")
    for item in candidates[:limit]:
        title = item.get("title", "")
        short_title = title if len(title) <= 55 else title[:52] + "..."
        print(
            f"{item.get('candidate_id'):>2} | {item.get('topic', ''):<12} | "
            f"{item.get('priority_score', 0):>4} | {short_title}"
        )
    print("")


def publish_selected_news(
    selected_news: list[dict],
    target_dates: set,
    source_label: str,
    preserve_selection: bool = False,
    max_items: int | None = 10,
) -> str:
    publish_items = list(selected_news)
    if not preserve_selection:
        publish_items = enforce_selected_rules(publish_items, max_per_keyword=3)
    if max_items is not None:
        publish_items = publish_items[: max(1, int(max_items))]

    top10_path = save_top10_csv(publish_items, TOP10_FILE)
    pdf_name = f"Daily_News_Monitoring_{datetime.now(KST):%Y%m%d}.pdf"
    pdf_path = build_pdf_for_selected(publish_items, pdf_name)
    sent = send_email_with_pdf(pdf_path, len(publish_items), target_dates)
    email_state = "sent" if sent else "not sent"
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"Selected Items: {top10_path} | PDF: {pdf_path} | Email: {email_state}"
    )
    print(f"Selected IDs from {source_label}: {', '.join(str(i['candidate_id']) for i in publish_items)}")
    return pdf_path


def interactive_publish_flow() -> None:
    candidates, target_dates = collect_candidates()
    candidates_path = save_candidates_to_csv(candidates, CANDIDATES_FILE)
    if not candidates:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No candidates found.")
        return

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Candidates saved: {candidates_path} "
        f"({len(candidates)} items)."
    )
    print_candidate_preview(candidates, limit=min(20, len(candidates)))

    raw = input("발행할 candidate_id를 입력하세요 (예: 1,3,8). 엔터만 누르면 상위 10건: ").strip()
    if raw:
        selected_ids = parse_ids_input(raw)
    else:
        selected_ids = [item.get("candidate_id") for item in candidates[:10]]

    selected_news = pick_selected_news(candidates, selected_ids)
    if not selected_news:
        print("선택한 ID가 후보군과 일치하지 않습니다.")
        return

    publish_selected_news(selected_news, target_dates, candidates_path)


def draw_page_number(c, page_number: int, total_pages: int, page_width: float, bottom_margin: float, font_name: str) -> None:
    c.setFont(font_name, 10)
    c.drawCentredString(page_width / 2, bottom_margin / 2, f"{page_number}/{total_pages}")


def wrap_text_for_canvas(c, text: str, max_width: float, font_name: str, font_size: int) -> list[str]:
    c.setFont(font_name, font_size)
    wrapped_lines = []
    for paragraph in text.split("\n"):
        para = paragraph.strip()
        if not para:
            wrapped_lines.append("")
            continue

        current = ""
        for ch in para:
            candidate = current + ch
            if c.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
                continue
            if current:
                wrapped_lines.append(current)
            current = ch

        if current:
            wrapped_lines.append(current)
    return wrapped_lines


def consume_lines_for_height(lines: list[str], max_height: float) -> int:
    used = 0.0
    count = 0
    for line in lines:
        line_height = 8 if not line else 14
        if used + line_height > max_height:
            break
        used += line_height
        count += 1
    return count


def build_pdf_for_selected(news_items: list[dict], file_path: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    safe_path = make_safe_output_path(file_path)
    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    regular_path = Path("C:/Windows/Fonts/malgun.ttf")
    bold_path = Path("C:/Windows/Fonts/malgunbd.ttf")
    if regular_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("Malgun", str(regular_path)))
        pdfmetrics.registerFont(TTFont("Malgun-Bold", str(bold_path)))
        regular_font = "Malgun"
        bold_font = "Malgun-Bold"
    else:
        # Linux/Cloud fallback for Korean text rendering.
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
            regular_font = "HYSMyeongJo-Medium"
            bold_font = "HYSMyeongJo-Medium"
        except Exception:
            pass

    page_w, page_h = A4
    margin = 2 * cm
    c = canvas.Canvas(safe_path, pagesize=A4)
    content_width = page_w - (2 * margin)
    value_x = margin + 50
    value_width = page_w - margin - value_x

    # Pre-compute wrapped content and required page count per article.
    article_blocks = []
    continuation_top_gap = 24
    for item in news_items:
        body_text = str(item.get("body_override", "") or "").strip() or fetch_article_body(item.get("link", ""))
        summary_note_text = item.get("summary_note", "") or ""
        title_lines = wrap_text_for_canvas(c, item.get("title", ""), content_width, bold_font, 16)[:3]
        link_lines = wrap_text_for_canvas(c, item.get("link", ""), value_width, regular_font, 11)[:2]
        summary_lines = wrap_text_for_canvas(c, summary_note_text, content_width, regular_font, 11) if summary_note_text else []
        body_lines = wrap_text_for_canvas(c, body_text, content_width, regular_font, 11)

        y = page_h - margin
        y -= 20 * len(title_lines)
        y -= 4
        y -= 16  # published
        y -= 16  # media
        y -= 16  # topic
        y -= 14 * len(link_lines)  # link lines
        y -= 8
        if summary_lines:
            y -= 18  # summary title gap
            y -= 14 * len(summary_lines)
            y -= 8
        y -= 18  # body title gap

        first_page_height = max(0, y - (margin + 28))
        cont_page_height = max(0, (page_h - margin - continuation_top_gap) - (margin + 28))

        remaining = list(body_lines)
        pages_needed = 1
        take = consume_lines_for_height(remaining, first_page_height)
        remaining = remaining[take:]
        while remaining:
            take = consume_lines_for_height(remaining, cont_page_height)
            if take <= 0:
                break
            remaining = remaining[take:]
            pages_needed += 1

        article_blocks.append(
            {
                "item": item,
                "title_lines": title_lines,
                "link_lines": link_lines,
                "summary_lines": summary_lines,
                "body_lines": body_lines,
                "pages_needed": pages_needed,
                "first_page_height": first_page_height,
                "cont_page_height": cont_page_height,
            }
        )

    total_pages = 2 + sum(block["pages_needed"] for block in article_blocks)

    today_text = datetime.now(KST).strftime("%Y/%m/%d")
    c.setFont(bold_font, 30)
    c.drawCentredString(page_w / 2, page_h / 2 + 20, f"Daily News Monitoring- {today_text}")
    c.setFont(regular_font, 13)
    c.drawCentredString(page_w / 2, page_h / 2 - 16, "BNP Paribas Cardif Life")
    draw_page_number(c, 1, total_pages, page_w, margin, regular_font)
    c.showPage()

    # TOC page
    c.setFont(bold_font, 22)
    c.drawString(margin, page_h - margin, "목 차")
    y = page_h - margin - 30
    c.setFont(regular_font, 11)
    next_article_page = 3
    for idx, block in enumerate(article_blocks, start=1):
        article_page = next_article_page
        next_article_page += block["pages_needed"]
        title = block["item"].get("title", "")
        toc_lines = wrap_text_for_canvas(c, title, page_w - (2 * margin) - 45, regular_font, 11)
        if not toc_lines:
            toc_lines = ["(제목 없음)"]

        first_line = f"{idx}. {toc_lines[0]}"
        if y < margin + 36:
            break
        c.drawString(margin, y, first_line)
        c.drawRightString(page_w - margin, y, str(article_page))
        y -= 14

        for extra_line in toc_lines[1:2]:
            if y < margin + 36:
                break
            c.drawString(margin + 18, y, extra_line)
            y -= 12

        y -= 2

    draw_page_number(c, 2, total_pages, page_w, margin, regular_font)
    c.showPage()

    current_page = 3
    for idx, block in enumerate(article_blocks, start=1):
        item = block["item"]
        body_lines = list(block["body_lines"])
        y = page_h - margin

        c.setFont(bold_font, 16)
        for line in block["title_lines"]:
            c.drawString(margin, y, line)
            y -= 20

        y -= 4
        c.setFont(bold_font, 11)
        c.drawString(margin, y, "발행일")
        c.setFont(regular_font, 11)
        c.drawString(margin + 50, y, item.get("published_at", ""))
        y -= 16

        c.setFont(bold_font, 11)
        c.drawString(margin, y, "매체")
        c.setFont(regular_font, 11)
        c.drawString(margin + 50, y, item.get("media_domain", ""))
        y -= 16

        c.setFont(bold_font, 11)
        c.drawString(margin, y, "분류")
        c.setFont(regular_font, 11)
        c.drawString(margin + 50, y, item.get("priority_terms", ""))
        y -= 16

        c.setFont(bold_font, 11)
        c.drawString(margin, y, "원문")
        c.setFont(regular_font, 11)
        for line in block["link_lines"]:
            c.drawString(value_x, y, line)
            y -= 14

        if block["summary_lines"]:
            y -= 8
            c.setFont(bold_font, 12)
            c.drawString(margin, y, "요약")
            y -= 18
            c.setFont(regular_font, 11)
            for line in block["summary_lines"]:
                c.drawString(margin, y, line)
                y -= 14

        y -= 8
        c.setFont(bold_font, 12)
        c.drawString(margin, y, "본문")
        y -= 18

        c.setFont(regular_font, 11)
        first_take = consume_lines_for_height(body_lines, block["first_page_height"])
        first_chunk = body_lines[:first_take]
        remaining = body_lines[first_take:]
        for line in first_chunk:
            if not line:
                y -= 8
            else:
                c.drawString(margin, y, line)
                y -= 14

        draw_page_number(c, current_page, total_pages, page_w, margin, regular_font)
        c.showPage()
        current_page += 1

        # Continuation pages for long articles.
        while remaining:
            y = page_h - margin
            c.setFont(bold_font, 13)
            c.drawString(margin, y, f"{item.get('title', '')} (계속)")
            y -= continuation_top_gap

            c.setFont(regular_font, 11)
            take = consume_lines_for_height(remaining, block["cont_page_height"])
            chunk = remaining[:take]
            remaining = remaining[take:]
            for line in chunk:
                if not line:
                    y -= 8
                else:
                    c.drawString(margin, y, line)
                    y -= 14

            draw_page_number(c, current_page, total_pages, page_w, margin, regular_font)
            c.showPage()
            current_page += 1

    c.save()
    return safe_path


def send_email_with_pdf(pdf_path: str, selected_count: int, target_dates: set) -> bool:
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not EMAIL_FROM or not EMAIL_TO:
        print("Email settings are empty. Skipping email send.")
        return False

    date_range_text = ", ".join(sorted(d.strftime("%Y-%m-%d") for d in target_dates))
    msg = EmailMessage()
    msg["Subject"] = f"[뉴스모니터링] Cardif Selected News ({date_range_text})"
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    msg.set_content(
        f"카디프생명 뉴스모니터링 선정 기사 PDF를 첨부합니다.\n"
        f"대상 기간: {date_range_text}\n"
        f"선정 기사 수: {selected_count}건\n"
    )

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    mime_type, _ = mimetypes.guess_type(pdf_path)
    maintype, subtype = (mime_type or "application/pdf").split("/", 1)
    msg.add_attachment(pdf_data, maintype=maintype, subtype=subtype, filename=Path(pdf_path).name)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
    return True


def collect_candidates(lookback_days: int | None = None, max_candidates: int | None = None) -> tuple[list[dict], set]:
    now_kst = datetime.now(KST)
    days_back = max(1, int(lookback_days if lookback_days is not None else _effective_days_back_from_config()))
    scoring_cfg = copy.deepcopy(NEWS_MONITORING_CONFIG)
    scoring_cfg.setdefault("scoring", {})
    scoring_cfg["scoring"]["focus_weights"] = copy.deepcopy(FOCUS_WEIGHTS)

    query_specs: list[tuple[str, str]] = []
    seen_query_texts: set[str] = set()

    # Primary query source: YAML query groups.
    for group in NEWS_MONITORING_CONFIG.get("query_groups", []):
        query = str(group.get("query", "")).strip()
        group_key = str(group.get("key", "")).strip() or "YAML_QUERY"
        if not query or query in seen_query_texts:
            continue
        query_specs.append((query, group_key))
        seen_query_texts.add(query)

    all_batches = []
    for query, group_key in query_specs:
        try:
            all_batches.append(fetch_news_by_keyword(query, group_key))
        except Exception as exc:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Failed query '{query}': {exc}")

    merged = merge_and_deduplicate(all_batches)
    by_date = filter_by_days_back(merged, now_kst, days_back)
    no_sensitive = [item for item in by_date if not contains_sensitive_topic(item)]
    no_sports_noise = [item for item in no_sensitive if not is_sports_noise_article(item)]
    pool = [score_article(item, scoring_cfg, KEYWORDS_CONFIG, now_kst=now_kst) for item in no_sports_noise]
    deduped_scored = dedupe_and_cluster(pool, scoring_cfg)
    ranked = [row.item for row in deduped_scored]
    target_count = max(1, min(100, max_candidates if max_candidates is not None else CANDIDATE_COUNT))
    quota_selected = apply_topic_quotas(ranked, target_count)
    selected_keys = {item.get("link") or item.get("title") for item in quota_selected}
    ordered_pool = quota_selected + [item for item in ranked if (item.get("link") or item.get("title")) not in selected_keys]

    candidates = select_candidates_with_fast_validation(ordered_pool, target_count)

    candidates = sorted(
        candidates,
        key=lambda x: (
            LEVEL_ORDER.get(x.get("level", "NOISE"), 9),
            -(x.get("score", 0)),
            -(x.get("published_at_dt").timestamp() if x.get("published_at_dt") else 0),
        ),
    )
    candidates = enforce_topic_cap(candidates, max_per_topic=20)
    candidates = enforce_keyword_media_cap(candidates, max_per_pair=2)

    for idx, item in enumerate(candidates, start=1):
        item["candidate_id"] = idx
    target_dates = {item["published_at_dt"].date() for item in candidates if item.get("published_at_dt")}
    return candidates, target_dates


def job() -> None:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("Please set NAVER_CLIENT_ID and NAVER_CLIENT_SECRET in the script.")
        return

    try:
        publish_source = find_candidates_file_for_publish(CANDIDATES_FILE)
        selected_news = read_selected_from_candidates(publish_source)
        if selected_news:
            target_dates = get_target_dates(datetime.now(KST))
            publish_selected_news(selected_news, target_dates, publish_source)
            return

        candidates, _ = collect_candidates()
        candidates_path = save_candidates_to_csv(candidates, CANDIDATES_FILE)
        if not candidates:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No candidates found.")
            return
        if len(candidates) < 30:
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Candidates saved: {candidates_path} "
                f"({len(candidates)} items, less than requested 30 due to filters)."
            )
            print("Set Y in the rightmost 'publish_yn' column for rows to publish, then run again.")
            return

        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Candidates saved: {candidates_path} "
            f"({len(candidates)} items)."
        )
        print("Set Y in the rightmost 'publish_yn' column for rows to publish, then run again.")
    except Exception as exc:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Job failed: {exc}")


def main() -> None:
    load_settings()
    if RUN_ONCE_FOR_TEST:
        print("Running monitoring job once now (interactive selection mode).")
        interactive_publish_flow()
        return

    schedule.every().day.at(SCHEDULE_TIME).do(job)
    print(f"Scheduler started. News monitoring runs every day at {SCHEDULE_TIME}.")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()

