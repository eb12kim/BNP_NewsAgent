from __future__ import annotations

from pathlib import Path

import streamlit as st

import main

TOPIC_LABELS = {
    "company_group": "당사/그룹",
    "policy_reg": "정책/감독",
    "lending": "대출",
    "els": "ELS 이슈",
    "product": "상품",
    "other": "기타",
}

PRESET_LABELS = {
    "balanced": "균형형",
    "company_first": "당사 우선",
    "policy_first": "정책 우선",
    "els_lending_first": "ELS/대출 우선",
}


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _apply_secret_credentials() -> None:
    # Streamlit Cloud: set these in App settings -> Secrets.
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
    if "selection_map" not in st.session_state:
        st.session_state.selection_map = {}
    if "last_candidates_csv" not in st.session_state:
        st.session_state.last_candidates_csv = ""
    if "last_pdf_path" not in st.session_state:
        st.session_state.last_pdf_path = ""


def _preset_key_from_label(label: str) -> str:
    for key, value in PRESET_LABELS.items():
        if value == label:
            return key
    return "balanced"


def _build_settings(
    preset_label: str,
    lookback_days: int,
    candidate_count: int,
    w_company: float,
    w_policy: float,
    w_els: float,
    output_dir: str,
) -> dict:
    preset_key = _preset_key_from_label(preset_label)
    return main.build_settings_with_preset(
        preset=preset_key,
        lookback_days=int(lookback_days),
        candidate_count=int(candidate_count),
        focus_weights={
            "company_group": float(w_company),
            "policy_reg": float(w_policy),
            "els_lending": float(w_els),
        },
        output_dir=output_dir,
    )


def _build_table_rows(candidates: list[dict], selected_ids: set[int]) -> list[dict]:
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


def app() -> None:
    st.set_page_config(page_title="BNP News Monitoring", layout="wide")
    _apply_secret_credentials()
    _init_state()
    current = st.session_state.settings
    current_lookback = _clamp_int(int(current.get("lookback_days", 3)), 1, 30)
    current_candidate_count = _clamp_int(int(current.get("candidate_count", 100)), 10, 100)

    st.title("Daily News Monitoring")
    if not main.NAVER_CLIENT_ID or not main.NAVER_CLIENT_SECRET:
        st.error(
            "NAVER API 키가 없습니다. Streamlit Cloud Secrets 또는 환경변수에 "
            "`NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`를 설정하세요."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    with c1:
        preset_label = st.selectbox(
            "프리셋",
            options=list(PRESET_LABELS.values()),
            index=list(PRESET_LABELS.keys()).index(current.get("preset", "balanced")),
        )
    with c2:
        lookback_days = st.number_input(
            "조회기간(일)", min_value=1, max_value=30, value=current_lookback, step=1
        )
    with c3:
        candidate_count = st.number_input(
            "후보수(최대100)", min_value=10, max_value=100, value=current_candidate_count, step=1
        )
    with c4:
        output_dir = st.text_input("산출물 저장 경로", value=str(current.get("output_dir", ".")))

    w1, w2, w3 = st.columns(3)
    with w1:
        w_company = st.slider(
            "당사/그룹 가중치",
            min_value=0.5,
            max_value=2.0,
            value=float(current.get("focus_weights", {}).get("company_group", 1.0)),
            step=0.1,
        )
    with w2:
        w_policy = st.slider(
            "정책/감독 가중치",
            min_value=0.5,
            max_value=2.0,
            value=float(current.get("focus_weights", {}).get("policy_reg", 1.0)),
            step=0.1,
        )
    with w3:
        w_els = st.slider(
            "ELS/대출 가중치",
            min_value=0.5,
            max_value=2.0,
            value=float(current.get("focus_weights", {}).get("els_lending", 1.0)),
            step=0.1,
        )

    settings = _build_settings(
        preset_label=preset_label,
        lookback_days=int(lookback_days),
        candidate_count=int(candidate_count),
        w_company=float(w_company),
        w_policy=float(w_policy),
        w_els=float(w_els),
        output_dir=output_dir,
    )

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("설정 저장", use_container_width=True):
            main.apply_runtime_settings(settings)
            main.save_settings(settings)
            st.session_state.settings = settings
            st.success("설정을 저장했습니다.")
    with b2:
        if st.button("후보 생성", use_container_width=True):
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
            st.session_state.selection_map = {int(i.get("candidate_id", 0)): False for i in candidates}
            st.session_state.last_candidates_csv = csv_path
            st.success(f"후보 {len(candidates)}건 생성 완료")

    if st.session_state.last_candidates_csv:
        st.caption(f"후보 CSV: {st.session_state.last_candidates_csv}")

    candidates = st.session_state.candidates
    if candidates:
        selected_ids = {cid for cid, checked in st.session_state.selection_map.items() if checked}
        rows = _build_table_rows(candidates, selected_ids)

        edited = st.data_editor(
            rows,
            use_container_width=True,
            hide_index=True,
            key="candidate_editor",
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
                "원문": st.column_config.LinkColumn(display_text="바로가기"),
            },
            disabled=["ID", "레벨", "토픽", "점수", "발행시각", "매체", "제목", "원문"],
        )

        new_selection_map: dict[int, bool] = {}
        for row in edited:
            cid = int(row.get("ID", 0) or 0)
            new_selection_map[cid] = bool(row.get("선택", False))
        st.session_state.selection_map = new_selection_map

        if st.button("선택 기사 PDF 생성", type="primary"):
            ids = sorted([cid for cid, checked in st.session_state.selection_map.items() if checked])
            if not ids:
                st.warning("PDF로 생성할 기사를 먼저 선택하세요.")
            else:
                selected_news = main.pick_selected_news(candidates, ids)
                with st.spinner("PDF 생성 중입니다..."):
                    pdf_path = main.publish_selected_news(
                        selected_news,
                        st.session_state.target_dates,
                        "Streamlit selection",
                        preserve_selection=True,
                        max_items=None,
                    )
                st.session_state.last_pdf_path = pdf_path
                st.success(f"PDF 생성 완료: {pdf_path}")

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
