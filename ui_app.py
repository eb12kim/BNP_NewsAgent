import copy
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

import main


PRESET_LABELS = {
    "balanced": "Balanced",
    "company_first": "Company First",
    "policy_first": "Policy First",
    "els_lending_first": "ELS/Lending First",
}

TOPIC_LABELS = {
    "company_group": "Company/Group",
    "policy_reg": "Policy/Reg",
    "lending": "Lending",
    "els": "ELS",
    "product": "Product",
    "other": "Other",
}


class DraftEditor(tk.Toplevel):
    def __init__(self, master: tk.Tk, selected_items: list[dict]) -> None:
        super().__init__(master)
        self.title("Draft Editor")
        self.geometry("1100x700")
        self.saved_items: list[dict] | None = None
        self.items: list[dict] = [copy.deepcopy(item) for item in selected_items]
        self.current_index: int | None = None
        self.body_cache: dict[str, str] = {}

        self._build_widgets()
        self.refresh_list()
        if self.items:
            self.select_index(0)

    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=False)

        ttk.Label(left, text="Selected Articles (order)").pack(anchor="w")
        cols = ("order", "id", "score", "title")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=28)
        self.tree.heading("order", text="#")
        self.tree.heading("id", text="ID")
        self.tree.heading("score", text="Score")
        self.tree.heading("title", text="Title")
        self.tree.column("order", width=45, anchor="center")
        self.tree.column("id", width=55, anchor="center")
        self.tree.column("score", width=70, anchor="center")
        self.tree.column("title", width=420, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(4, 8))
        self.tree.bind("<<TreeviewSelect>>", self.on_select_item)

        order_frame = ttk.Frame(left)
        order_frame.pack(fill="x")
        ttk.Button(order_frame, text="Move Up", command=self.move_up).pack(side="left", padx=(0, 6))
        ttk.Button(order_frame, text="Move Down", command=self.move_down).pack(side="left", padx=(0, 6))
        ttk.Button(order_frame, text="Remove", command=self.remove_current).pack(side="left", padx=(0, 6))
        ttk.Button(order_frame, text="Open Link", command=self.open_link).pack(side="left")

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(right, text="Edited Title").pack(anchor="w")
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(right, textvariable=self.title_var)
        self.title_entry.pack(fill="x", pady=(2, 8))

        ttk.Label(right, text="Summary Note (shown in PDF)").pack(anchor="w")
        self.summary_text = tk.Text(right, height=8, wrap="word")
        self.summary_text.pack(fill="x", expand=False, pady=(2, 8))

        ttk.Label(right, text="Article Body (editable, shown in PDF)").pack(anchor="w")
        self.body_text = tk.Text(right, height=20, wrap="word")
        self.body_text.pack(fill="both", expand=True, pady=(2, 8))

        actions = ttk.Frame(right)
        actions.pack(fill="x")
        ttk.Button(actions, text="Apply", command=self.apply_current).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Apply + Next", command=self.apply_and_next).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Save Draft", command=self.save_and_close).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right")

    def refresh_list(self) -> None:
        for rid in self.tree.get_children():
            self.tree.delete(rid)
        for idx, item in enumerate(self.items, start=1):
            title = (item.get("title", "") or "").strip()
            short = title if len(title) <= 70 else f"{title[:67]}..."
            self.tree.insert(
                "",
                "end",
                iid=str(idx - 1),
                values=(
                    idx,
                    int(item.get("candidate_id", 0)),
                    int(item.get("score", item.get("priority_score", 0)) or 0),
                    short,
                ),
            )

    def on_select_item(self, _) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        self.load_index(int(selected[0]))

    def load_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.items):
            return
        self.current_index = idx
        item = self.items[idx]
        self.title_var.set(str(item.get("title", "") or ""))
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", str(item.get("summary_note", "") or ""))
        self.body_text.delete("1.0", "end")
        body_override = str(item.get("body_override", "") or "").strip()
        if body_override:
            self.body_text.insert("1.0", body_override)
            return

        link = str(item.get("link", "") or "").strip()
        if not link:
            return
        if link in self.body_cache:
            body = self.body_cache[link]
        else:
            self.body_text.insert("1.0", "Loading article body...")
            self.update_idletasks()
            body = main.fetch_article_body(link)
            self.body_cache[link] = body
        self.body_text.delete("1.0", "end")
        self.body_text.insert("1.0", body)

    def select_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.items):
            return
        self.tree.selection_set(str(idx))
        self.tree.focus(str(idx))
        self.tree.see(str(idx))
        self.load_index(idx)

    def apply_current(self) -> None:
        if self.current_index is None:
            return
        item = self.items[self.current_index]
        item["title"] = self.title_var.get().strip()
        item["summary_note"] = self.summary_text.get("1.0", "end").strip()
        item["body_override"] = self.body_text.get("1.0", "end").strip()
        self.refresh_list()
        self.select_index(self.current_index)

    def apply_and_next(self) -> None:
        if self.current_index is None:
            return
        self.apply_current()
        next_idx = min(self.current_index + 1, len(self.items) - 1)
        self.select_index(next_idx)

    def move_up(self) -> None:
        if self.current_index is None or self.current_index <= 0:
            return
        idx = self.current_index
        self.items[idx - 1], self.items[idx] = self.items[idx], self.items[idx - 1]
        self.refresh_list()
        self.select_index(idx - 1)

    def move_down(self) -> None:
        if self.current_index is None or self.current_index >= len(self.items) - 1:
            return
        idx = self.current_index
        self.items[idx + 1], self.items[idx] = self.items[idx], self.items[idx + 1]
        self.refresh_list()
        self.select_index(idx + 1)

    def remove_current(self) -> None:
        if self.current_index is None:
            return
        idx = self.current_index
        title = str(self.items[idx].get("title", "") or "").strip()
        short_title = title if len(title) <= 50 else f"{title[:47]}..."
        if not messagebox.askyesno("Remove", f"Delete this article from draft?\n\n{short_title}", parent=self):
            return
        self.items.pop(idx)
        self.refresh_list()
        if not self.items:
            self.current_index = None
            self.title_var.set("")
            self.summary_text.delete("1.0", "end")
            self.body_text.delete("1.0", "end")
            return
        next_idx = min(idx, len(self.items) - 1)
        self.select_index(next_idx)

    def open_link(self) -> None:
        if self.current_index is None:
            return
        link = str(self.items[self.current_index].get("link", "")).strip()
        if link:
            webbrowser.open(link, new=2)

    def save_and_close(self) -> None:
        self.apply_current()
        self.saved_items = self.items
        self.destroy()


class NewsMonitorUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Daily News Monitoring")
        self.root.geometry("1360x820")

        self.candidates: list[dict] = []
        self.candidate_map: dict[int, dict] = {}
        self.target_dates: set = set()
        self.checked_ids: set[int] = set()
        self.draft_items: list[dict] = []

        self.settings = main.load_settings()
        self._build_widgets()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Preset").pack(side="left")
        self.preset_var = tk.StringVar(value=self.settings.get("preset", "balanced"))
        self.preset_combo = ttk.Combobox(top, state="readonly", width=16, values=list(PRESET_LABELS.values()))
        self.preset_combo.pack(side="left", padx=(4, 10))
        self._set_preset_combo_from_key(self.preset_var.get())
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_change)

        ttk.Label(top, text="Lookback(days)").pack(side="left")
        self.lookback_var = tk.IntVar(value=max(1, int(self.settings.get("lookback_days", 3))))
        tk.Spinbox(top, from_=1, to=30, width=5, textvariable=self.lookback_var).pack(side="left", padx=(4, 10))

        ttk.Label(top, text="Candidates(max100)").pack(side="left")
        self.candidate_count_var = tk.IntVar(value=min(100, int(self.settings.get("candidate_count", 100))))
        tk.Spinbox(top, from_=10, to=100, width=5, textvariable=self.candidate_count_var).pack(side="left", padx=(4, 10))

        ttk.Label(top, text="Company").pack(side="left")
        self.w_company = tk.DoubleVar(value=float(self.settings.get("focus_weights", {}).get("company_group", 1.0)))
        tk.Scale(top, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", length=90, variable=self.w_company).pack(side="left")

        ttk.Label(top, text="Policy").pack(side="left")
        self.w_policy = tk.DoubleVar(value=float(self.settings.get("focus_weights", {}).get("policy_reg", 1.0)))
        tk.Scale(top, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", length=90, variable=self.w_policy).pack(side="left")

        ttk.Label(top, text="ELS/Lending").pack(side="left")
        self.w_els = tk.DoubleVar(value=float(self.settings.get("focus_weights", {}).get("els_lending", 1.0)))
        tk.Scale(top, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", length=90, variable=self.w_els).pack(side="left")

        ttk.Label(top, text="Output").pack(side="left", padx=(8, 0))
        self.output_dir_var = tk.StringVar(value=str(self.settings.get("output_dir", ".")))
        ttk.Entry(top, textvariable=self.output_dir_var, width=28).pack(side="left", padx=(4, 4))
        ttk.Button(top, text="Browse", command=self.pick_output_dir).pack(side="left")

        btn = ttk.Frame(self.root, padding=(10, 0))
        btn.pack(fill="x")

        self.btn_fetch = ttk.Button(btn, text="1) Generate Candidates", command=self.fetch_candidates)
        self.btn_fetch.pack(side="left", padx=(0, 8))
        self.btn_edit = ttk.Button(btn, text="2) Edit Draft", command=self.open_draft_editor_from_selection)
        self.btn_edit.pack(side="left", padx=(0, 8))
        self.btn_publish = ttk.Button(btn, text="3) Build PDF", command=self.publish_from_draft)
        self.btn_publish.pack(side="left", padx=(0, 8))
        ttk.Button(btn, text="Select All", command=self.select_all).pack(side="left", padx=(0, 8))
        ttk.Button(btn, text="Clear", command=self.clear_selection).pack(side="left", padx=(0, 8))
        ttk.Button(btn, text="Save Settings", command=self.save_current_settings).pack(side="left", padx=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btn, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        cols = ("check", "id", "level", "topic", "score", "published", "media", "title", "link")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", selectmode="none")
        self.tree.heading("check", text="Sel")
        self.tree.heading("id", text="ID")
        self.tree.heading("level", text="Level")
        self.tree.heading("topic", text="Topic")
        self.tree.heading("score", text="Score")
        self.tree.heading("published", text="Published")
        self.tree.heading("media", text="Media")
        self.tree.heading("title", text="Title")
        self.tree.heading("link", text="Open")
        self.tree.column("check", width=50, anchor="center")
        self.tree.column("id", width=55, anchor="center")
        self.tree.column("level", width=95, anchor="center")
        self.tree.column("topic", width=120, anchor="center")
        self.tree.column("score", width=80, anchor="center")
        self.tree.column("published", width=150, anchor="center")
        self.tree.column("media", width=170, anchor="w")
        self.tree.column("title", width=550, anchor="w")
        self.tree.column("link", width=70, anchor="center")
        self.tree.tag_configure("checked", background="#dbeafe")
        self.tree.bind("<Button-1>", self._on_tree_click)

        table_frame = ttk.Frame(self.root)
        table_frame.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

    def _preset_key_from_label(self, label: str) -> str:
        for key, value in PRESET_LABELS.items():
            if value == label:
                return key
        return "balanced"

    def _set_preset_combo_from_key(self, preset_key: str) -> None:
        self.preset_combo.set(PRESET_LABELS.get(preset_key, PRESET_LABELS["balanced"]))

    def on_preset_change(self, _) -> None:
        preset_key = self._preset_key_from_label(self.preset_combo.get())
        preset = main.PRESET_SETTINGS.get(preset_key, main.PRESET_SETTINGS["balanced"])
        self.preset_var.set(preset_key)
        self.lookback_var.set(max(1, int(preset.get("lookback_days", 3))))
        self.candidate_count_var.set(int(preset.get("candidate_count", 100)))
        fw = preset.get("focus_weights", {})
        self.w_company.set(float(fw.get("company_group", 1.0)))
        self.w_policy.set(float(fw.get("policy_reg", 1.0)))
        self.w_els.set(float(fw.get("els_lending", 1.0)))

    def _build_current_settings(self) -> dict:
        preset_key = self._preset_key_from_label(self.preset_combo.get())
        return main.build_settings_with_preset(
            preset=preset_key,
            lookback_days=int(self.lookback_var.get()),
            candidate_count=min(100, int(self.candidate_count_var.get())),
            focus_weights={
                "company_group": float(self.w_company.get()),
                "policy_reg": float(self.w_policy.get()),
                "els_lending": float(self.w_els.get()),
            },
            output_dir=self.output_dir_var.get().strip(),
        )

    def pick_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get().strip() or ".")
        if selected:
            self.output_dir_var.set(selected)

    def save_current_settings(self) -> None:
        self.sync_settings_runtime_and_save()
        messagebox.showinfo("Saved", "Settings saved.")

    def sync_settings_runtime_and_save(self) -> None:
        settings = self._build_current_settings()
        main.apply_runtime_settings(settings)
        main.save_settings(settings)
        self.settings = settings

    def _on_tree_click(self, event) -> str | None:
        region = self.tree.identify("region", event.x, event.y)
        if region not in {"cell", "tree"}:
            return None
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return None
        col_id = self.tree.identify_column(event.x)

        if col_id == f"#{len(self.tree['columns'])}":
            try:
                cid = int(row_id)
                item = self.candidate_map.get(cid, {})
                link = str(item.get("link", "")).strip()
                if link:
                    webbrowser.open(link, new=2)
            except Exception:
                pass
            return "break"

        cid = int(row_id)
        if cid in self.checked_ids:
            self.checked_ids.remove(cid)
            self.tree.set(row_id, "check", "☐")
            self.tree.item(row_id, tags=())
        else:
            self.checked_ids.add(cid)
            self.tree.set(row_id, "check", "☑")
            self.tree.item(row_id, tags=("checked",))
        return "break"

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def fetch_candidates(self) -> None:
        self.btn_fetch.config(state="disabled")
        self.btn_edit.config(state="disabled")
        self.btn_publish.config(state="disabled")
        self.set_status("Generating candidates...")
        threading.Thread(target=self._fetch_candidates_worker, daemon=True).start()

    def _fetch_candidates_worker(self) -> None:
        try:
            self.sync_settings_runtime_and_save()
            settings = self.settings
            candidates, target_dates = main.collect_candidates(
                lookback_days=int(settings["lookback_days"]),
                max_candidates=min(100, int(settings["candidate_count"])),
            )
            path = main.save_candidates_to_csv(candidates, main.CANDIDATES_FILE)
            self.root.after(0, self._on_fetch_done, candidates, target_dates, path)
        except Exception as exc:
            self.root.after(0, self._on_error, f"Candidate generation failed: {exc}")

    def _on_fetch_done(self, candidates: list[dict], target_dates: set, path: str) -> None:
        self.candidates = candidates
        self.candidate_map = {int(item.get("candidate_id")): item for item in candidates if item.get("candidate_id")}
        self.target_dates = target_dates
        self.checked_ids.clear()
        self.draft_items = []

        for row_id in self.tree.get_children():
            self.tree.delete(row_id)

        for item in self.candidates:
            cid = int(item.get("candidate_id"))
            topic_key = str(item.get("topic", "other")).strip().lower()
            topic_label = TOPIC_LABELS.get(topic_key, topic_key if topic_key else "Other")
            self.tree.insert(
                "",
                "end",
                iid=str(cid),
                values=(
                    "☐",
                    cid,
                    item.get("level", "NOISE"),
                    topic_label,
                    item.get("score", item.get("priority_score", 0)),
                    item.get("published_at", ""),
                    item.get("press_domain", item.get("media_domain", "")),
                    item.get("title", ""),
                    "Open",
                ),
                tags=(),
            )

        self.set_status(f"Generated {len(candidates)} candidates ({path})")
        self.btn_fetch.config(state="normal")
        self.btn_edit.config(state="normal")
        self.btn_publish.config(state="normal")

    def _selected_news(self) -> list[dict]:
        if not self.candidates:
            return []
        if not self.checked_ids:
            return []
        return main.pick_selected_news(self.candidates, sorted(self.checked_ids))

    def open_draft_editor_from_selection(self) -> None:
        selected_news = self._selected_news()
        if not selected_news:
            messagebox.showwarning("Notice", "Select articles first.")
            return
        editor = DraftEditor(self.root, selected_news)
        self.root.wait_window(editor)
        if editor.saved_items is None:
            return
        self.draft_items = editor.saved_items
        path = main.save_draft_edits(self.draft_items, main.DRAFT_EDITS_FILE)
        self.set_status(f"Draft saved ({len(self.draft_items)} items): {path}")
        messagebox.showinfo("Saved", f"Draft saved.\n{path}")

    def publish_from_draft(self) -> None:
        if not self.candidates:
            messagebox.showwarning("Notice", "Run candidate generation first.")
            return
        if not self.draft_items:
            messagebox.showwarning("Notice", "Edit Draft first (step 2).")
            return

        self.btn_fetch.config(state="disabled")
        self.btn_edit.config(state="disabled")
        self.btn_publish.config(state="disabled")
        self.set_status("Building PDF from draft...")
        threading.Thread(target=self._publish_worker, args=(copy.deepcopy(self.draft_items),), daemon=True).start()

    def _publish_worker(self, draft_items: list[dict]) -> None:
        try:
            pdf_path = main.publish_selected_news(
                draft_items,
                self.target_dates,
                "UI draft",
                preserve_selection=True,
                max_items=None,
            )
            self.root.after(0, self._on_publish_done, pdf_path, len(draft_items))
        except Exception as exc:
            self.root.after(0, self._on_error, f"PDF build failed: {exc}")

    def _on_publish_done(self, pdf_path: str, count: int) -> None:
        self.set_status(f"Done: {count} draft items -> {pdf_path}")
        self.btn_fetch.config(state="normal")
        self.btn_edit.config(state="normal")
        self.btn_publish.config(state="normal")
        messagebox.showinfo("Done", f"PDF generated:\n{pdf_path}")

    def _on_error(self, msg: str) -> None:
        self.set_status("Error")
        self.btn_fetch.config(state="normal")
        self.btn_edit.config(state="normal")
        self.btn_publish.config(state="normal")
        messagebox.showerror("Error", msg)

    def select_all(self) -> None:
        self.checked_ids = {int(item.get("candidate_id")) for item in self.candidates}
        for row_id in self.tree.get_children():
            self.tree.set(row_id, "check", "☑")
            self.tree.item(row_id, tags=("checked",))

    def clear_selection(self) -> None:
        self.checked_ids.clear()
        for row_id in self.tree.get_children():
            self.tree.set(row_id, "check", "☐")
            self.tree.item(row_id, tags=())


def main_ui() -> None:
    root = tk.Tk()
    NewsMonitorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main_ui()
