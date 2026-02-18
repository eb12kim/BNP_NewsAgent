from __future__ import annotations

from pathlib import Path

import streamlit as st

import main


TOPIC_LABELS = {
    "company_group": "당사/그룹",
    "policy_reg": "정책/감독",
    "competitor": "경쟁사",
    "lending": "대출",
    "els": "ELS",
    "product": "상품",
    "partnership": "파트너십",
    "other": "기타",
}

PRESET_LABELS = {
    "balanced": "Balanced",
    "company_first": "Company First",
    "policy_first": "Policy First",
    "els_lending_first": "ELS/Lending First",
    "competitor_watch": "Competitor Watch",
    "partnership_watch": "Partnership Watch",
    "risk_fast_alert": "Risk Fast Alert",
}

def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _apply_secret_credentials() -> None:
    try:
        cid = str(st.secrets.get("NAVER_CLIENT_ID", "")).strip()
        csec = str(st.secrets.get("NAVER_CLIENT_SECRET", "")).strip()
    except Exception:
        cid = ""
        csec = ""
    if cid and not main.NAVER_CLIENT_ID:
        main.NAVER_CLIENT_ID = cid
    if csec and not main.NAVER_CLIENT_SECRET:
        main.NAVER_CLIENT_SECRET = csec


def _load_runtime_settings() -> dict:
    settings = main.load_settings()
    main.apply_runtime_settings(settings)
    return settings


def _init_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = _load_runtime_settings()
    if "candidates" not in st.session_state:
        st.session_state.candidates = []
    if "target_dates" not in st.session_state:
        st.session_state.target_dates = set()
    if "selection_rows" not in st.session_state:
        st.session_state.selection_rows = []
    if "draft_rows" not in st.session_state:
        st.session_state.draft_rows = []
    if "last_candidates_csv" not in st.session_state:
        st.session_state.last_candidates_csv = ""
    if "last_pdf_path" not in st.session_state:
        st.session_state.last_pdf_path = ""
    if "last_draft_json" not in st.session_state:
        st.session_state.last_draft_json = ""


def _preset_key_from_label(label: str) -> str:
    for key, value in PRESET_LABELS.items():
        if value == label:
            return key
    return "balanced"


def _build_settings(
    preset_label: str,
    lookback_days: int,
    candidate_count: int,
    output_dir: str,
) -> dict:
    preset_key = _preset_key_from_label(preset_label)
    return main.build_settings_with_preset(
        preset=preset_key,
        lookback_days=int(lookback_days),
        candidate_count=int(candidate_count),
        output_dir=output_dir,
    )


def _normalize_rows(edited_data) -> list[dict]:
    if isinstance(edited_data, list):
        return [dict(row) for row in edited_data]
    to_dict = getattr(edited_data, "to_dict", None)
    if callable(to_dict):
        return [dict(row) for row in edited_data.to_dict(orient="records")]
    return []


def _build_selection_rows(candidates: list[dict], selected_ids: set[int] | None = None) -> list[dict]:
    selected_ids = selected_ids or set()
    rows = []
    for item in candidates:
        cid = int(item.get("candidate_id", 0) or 0)
        topic_key = str(item.get("topic", "other")).strip().lower() or "other"
        rows.append(
            {
                "선택": cid in selected_ids,
                "ID": cid,
                "레벨": item.get("level", "NOISE"),
                "토픽": TOPIC_LABELS.get(topic_key, topic_key),
                "점수": int(item.get("score", item.get("priority_score", 0)) or 0),
                "발행시각": item.get("published_at", ""),
                "매체": item.get("press_domain", item.get("media_domain", "")),
                "제목": item.get("title", ""),
                "원문": item.get("link", ""),
            }
        )
    return rows


def _draft_from_selected(selected_news: list[dict]) -> list[dict]:
    rows = []
    for idx, item in enumerate(selected_news, start=1):
        body_override = str(item.get("body_override", "") or "").strip()
        if not body_override:
            body_override = main.fetch_article_body(item.get("link", ""))
        rows.append(
            {
                "유지": True,
                "순서": idx,
                "ID": int(item.get("candidate_id", 0) or 0),
                "점수": int(item.get("score", item.get("priority_score", 0)) or 0),
                "제목": str(item.get("title", "") or ""),
                "요약": str(item.get("summary_note", "") or ""),
                "본문": body_override,
                "링크": str(item.get("link", "") or ""),
                "발행시각": str(item.get("published_at", "") or ""),
                "매체": str(item.get("media_domain", item.get("press_domain", "")) or ""),
            }
        )
    return rows


def _draft_rows_to_publish_items(draft_rows: list[dict]) -> list[dict]:
    kept = [r for r in draft_rows if bool(r.get("유지", True))]
    kept = sorted(kept, key=lambda r: int(r.get("순서", 9999) or 9999))
    items = []
    for row in kept:
        items.append(
            {
                "candidate_id": int(row.get("ID", 0) or 0),
                "priority_score": int(row.get("점수", 0) or 0),
                "score": int(row.get("점수", 0) or 0),
                "title": str(row.get("제목", "") or ""),
                "summary_note": str(row.get("요약", "") or ""),
                "body_override": str(row.get("본문", "") or ""),
                "link": str(row.get("링크", "") or ""),
                "published_at": str(row.get("발행시각", "") or ""),
                "media_domain": str(row.get("매체", "") or ""),
                "press_domain": str(row.get("매체", "") or ""),
            }
        )
    return items


def app() -> None:
    st.set_page_config(page_title="BNP News Monitoring", layout="wide")
    _apply_secret_credentials()
    _init_state()
    current = st.session_state.settings

    st.title("Daily News Monitoring (Streamlit)")
    if not main.NAVER_CLIENT_ID or not main.NAVER_CLIENT_SECRET:
        st.error("NAVER API 키가 없습니다. 환경변수 또는 Streamlit secrets에 설정하세요.")
        st.stop()

    current_lookback = _clamp_int(int(current.get("lookback_days", 3)), 1, 30)
    current_candidate_count = _clamp_int(int(current.get("candidate_count", 100)), 10, 100)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    preset_keys = list(PRESET_LABELS.keys())
    current_preset = current.get("preset", "balanced")
    preset_index = preset_keys.index(current_preset) if current_preset in preset_keys else 0
    with c1:
        preset_label = st.selectbox(
            "프리셋",
            options=list(PRESET_LABELS.values()),
            index=preset_index,
        )
    with c2:
        lookback_days = st.number_input("조회기간(일)", min_value=1, max_value=30, value=current_lookback, step=1)
    with c3:
        candidate_count = st.number_input("후보수(최대100)", min_value=10, max_value=100, value=current_candidate_count, step=1)
    with c4:
        output_dir = st.text_input("산출물 저장 경로", value=str(current.get("output_dir", ".")))

    settings = _build_settings(
        preset_label=preset_label,
        lookback_days=int(lookback_days),
        candidate_count=int(candidate_count),
        output_dir=output_dir,
    )

    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    with b1:
        if st.button("설정 저장", use_container_width=True):
            main.apply_runtime_settings(settings)
            main.save_settings(settings)
            st.session_state.settings = settings
            st.success("설정을 저장했습니다.")
    with b2:
        if st.button("1) 후보 생성", use_container_width=True):
            main.apply_runtime_settings(settings)
            main.save_settings(settings)
            st.session_state.settings = settings
            with st.spinner("후보 기사를 수집하는 중입니다..."):
                candidates, target_dates = main.collect_candidates(
                    lookback_days=int(settings["lookback_days"]),
                    max_candidates=min(100, int(settings["candidate_count"])),
                )
                csv_path = main.save_candidates_to_csv(candidates, main.CANDIDATES_FILE)
            st.session_state.candidates = candidates
            st.session_state.target_dates = target_dates
            st.session_state.selection_rows = _build_selection_rows(candidates, set())
            st.session_state.draft_rows = []
            st.session_state.last_candidates_csv = csv_path
            st.success(f"후보 {len(candidates)}건 생성 완료")
    with b3:
        if st.button("Draft 불러오기", use_container_width=True):
            loaded = main.load_draft_edits(main.DRAFT_EDITS_FILE)
            if not loaded:
                st.warning("저장된 draft가 없습니다.")
            else:
                draft_rows = []
                for idx, item in enumerate(loaded, start=1):
                    body_override = str(item.get("body_override", "") or "").strip()
                    if not body_override:
                        body_override = main.fetch_article_body(item.get("link", ""))
                    draft_rows.append(
                        {
                            "유지": True,
                            "순서": idx,
                            "ID": int(item.get("candidate_id", 0) or 0),
                            "점수": int(item.get("priority_score", item.get("score", 0)) or 0),
                            "제목": str(item.get("title", "") or ""),
                            "요약": str(item.get("summary_note", "") or ""),
                            "본문": body_override,
                            "링크": str(item.get("link", "") or ""),
                            "발행시각": str(item.get("published_at", "") or ""),
                            "매체": str(item.get("media_domain", "") or ""),
                        }
                    )
                st.session_state.draft_rows = draft_rows
                st.success(f"Draft {len(draft_rows)}건 불러오기 완료")

    if st.session_state.last_candidates_csv:
        st.caption(f"후보 CSV: {st.session_state.last_candidates_csv}")

    candidates = st.session_state.candidates
    st.subheader("후보 선택")
    if candidates:
        if not st.session_state.selection_rows or len(st.session_state.selection_rows) != len(candidates):
            selected_ids = {int(r.get("ID", 0) or 0) for r in st.session_state.selection_rows if bool(r.get("선택", False))}
            st.session_state.selection_rows = _build_selection_rows(candidates, selected_ids)
        with st.form("candidate_selection_form"):
            edited_sel = st.data_editor(
                st.session_state.selection_rows,
                use_container_width=True,
                hide_index=True,
                key="candidate_selector",
                num_rows="fixed",
                column_config={
                    "선택": st.column_config.CheckboxColumn(required=True),
                    "ID": st.column_config.NumberColumn(disabled=True),
                    "레벨": st.column_config.TextColumn(disabled=True),
                    "토픽": st.column_config.TextColumn(disabled=True),
                    "점수": st.column_config.NumberColumn(disabled=True),
                    "발행시각": st.column_config.TextColumn(disabled=True),
                    "매체": st.column_config.TextColumn(disabled=True),
                    "제목": st.column_config.TextColumn(disabled=True),
                    "원문": st.column_config.LinkColumn(display_text="열기"),
                },
                disabled=["ID", "레벨", "토픽", "점수", "발행시각", "매체", "제목", "원문"],
            )
            submit_selection = st.form_submit_button("2) 선택 항목을 Draft로 가져오기", use_container_width=True)

        rows = _normalize_rows(edited_sel)
        if rows:
            st.session_state.selection_rows = rows

        if submit_selection:
            ids = sorted([int(r.get("ID", 0) or 0) for r in st.session_state.selection_rows if bool(r.get("선택", False))])
            if not ids:
                st.warning("먼저 후보를 선택하세요.")
            else:
                selected_news = main.pick_selected_news(candidates, ids)
                with st.spinner("선택 기사 본문을 불러와 Draft를 구성하는 중입니다..."):
                    st.session_state.draft_rows = _draft_from_selected(selected_news)
                st.success(f"Draft {len(st.session_state.draft_rows)}건 준비 완료")
    else:
        st.info("후보를 먼저 생성하세요.")

    st.subheader("Draft 편집 (제목/요약/본문/삭제/순서)")
    if st.session_state.draft_rows:
        edited_draft = st.data_editor(
            st.session_state.draft_rows,
            use_container_width=True,
            hide_index=True,
            key="draft_editor",
            num_rows="fixed",
            height=520,
            column_config={
                "유지": st.column_config.CheckboxColumn(help="해제하면 PDF에서 제외됩니다."),
                "순서": st.column_config.NumberColumn(min_value=1, step=1),
                "ID": st.column_config.NumberColumn(disabled=True),
                "점수": st.column_config.NumberColumn(disabled=True),
                "제목": st.column_config.TextColumn(),
                "요약": st.column_config.TextColumn(),
                "본문": st.column_config.TextColumn(),
                "링크": st.column_config.LinkColumn(display_text="열기"),
                "발행시각": st.column_config.TextColumn(disabled=True),
                "매체": st.column_config.TextColumn(disabled=True),
            },
            disabled=["ID", "점수", "발행시각", "매체"],
        )
        rows = _normalize_rows(edited_draft)
        if rows:
            st.session_state.draft_rows = rows

        d1, d2, d3 = st.columns([1, 1, 2])
        with d1:
            if st.button("Draft 저장(JSON)", use_container_width=True):
                payload = _draft_rows_to_publish_items(st.session_state.draft_rows)
                draft_path = main.save_draft_edits(payload, main.DRAFT_EDITS_FILE)
                st.session_state.last_draft_json = draft_path
                st.success(f"저장 완료: {draft_path}")
        with d2:
            if st.button("3) Draft로 PDF 생성", type="primary", use_container_width=True):
                publish_items = _draft_rows_to_publish_items(st.session_state.draft_rows)
                if not publish_items:
                    st.warning("유지 항목이 없습니다. Draft에서 최소 1건 이상 유지하세요.")
                else:
                    with st.spinner("PDF 생성 중입니다..."):
                        pdf_path = main.publish_selected_news(
                            publish_items,
                            st.session_state.target_dates,
                            "Streamlit draft",
                            preserve_selection=True,
                            max_items=None,
                        )
                    st.session_state.last_pdf_path = pdf_path
                    st.success(f"PDF 생성 완료: {pdf_path}")
    else:
        st.info("선택 항목을 Draft로 가져오면 편집할 수 있습니다.")

    if st.session_state.last_draft_json:
        st.caption(f"Draft JSON: {st.session_state.last_draft_json}")

    if st.session_state.last_pdf_path:
        pdf_file = Path(st.session_state.last_pdf_path)
        if pdf_file.exists():
            with pdf_file.open("rb") as fh:
                st.download_button(
                    "PDF 다운로드",
                    data=fh.read(),
                    file_name=pdf_file.name,
                    mime="application/pdf",
                    use_container_width=True,
                )


if __name__ == "__main__":
    app()

