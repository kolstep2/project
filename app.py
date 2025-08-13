
# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import re
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber

NAME_HEADER_REGEX = re.compile(r"(наимен|описан|item|description)", re.IGNORECASE)

def page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count

def render_page_thumbnail(pdf_bytes: bytes, page_index: int, max_dim: int = 350) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        scale = min(max_dim / max(pix.width, pix.height), 1.0)
        if scale < 1.0:
            mat2 = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom) * mat2, alpha=False)
        return pix.tobytes("png")

def parse_page_ranges(s: str, total_pages: int) -> List[int]:
    pages: List[int] = []
    s = (s or "").strip()
    if not s:
        return pages
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                start = max(1, int(a))
                end = min(total_pages, int(b))
                if start <= end:
                    pages.extend(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.append(p)
            except ValueError:
                pass
    seen = set()
    result = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result

def extract_text_from_page(pdf_bytes: bytes, page_index0: int) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index0]
        return page.get_text("text") or ""

def extract_tables_with_name(pdf_bytes: bytes, page_numbers_1based: List[int]) -> Tuple[pd.DataFrame, List[Dict]]:
    rows = []
    raw_tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in page_numbers_1based:
            if p < 1 or p > len(pdf.pages):
                continue
            page = pdf.pages[p - 1]
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tbl in tables or []:
                if not tbl or not tbl[0]:
                    continue
                header = [ (h or "").strip() for h in tbl[0] ]
                name_idx: Optional[int] = None
                for idx, h in enumerate(header):
                    if NAME_HEADER_REGEX.search(h):
                        name_idx = idx
                        break
                raw_tables.append({"page": p, "header": header, "rows": tbl[1:]})
                if name_idx is None:
                    continue
                for r in tbl[1:]:
                    if not r:
                        continue
                    if name_idx < len(r):
                        name_val = (r[name_idx] or "").strip()
                        if name_val:
                            rows.append({
                                "Стр.": p,
                                "Наименование": name_val,
                                **{f"col_{i}": (r[i] if i < len(r) else None) for i in range(len(header))}
                            })
    df = pd.DataFrame(rows)
    return df, raw_tables

st.set_page_config(page_title="PDF → Наименование", layout="wide")
st.title("📄 PDF → выбор страниц → извлечение данных")

uploaded_file = st.file_uploader("Загрузите PDF", type=["pdf"]) 

if uploaded_file is None:
    st.info("Загрузите файл PDF, чтобы продолжить.")
    st.stop()

pdf_bytes: bytes = uploaded_file.read()
total = page_count(pdf_bytes)
st.caption(f"Определено страниц: **{total}**")

with st.expander("Выбор страниц", expanded=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Выберите страницы кликом по миниатюрам или введите диапазоны справа.")
        grid_cols = st.columns(6)
        selected = st.session_state.get("selected_pages", set())
        for i in range(min(total, 200)):
            thumb = render_page_thumbnail(pdf_bytes, i)
            with grid_cols[i % 6]:
                if st.button(key=f"thumb_{i}", label=f"Стр. {i+1}", help=f"Нажмите, чтобы (де)выбрать страницу {i+1}", use_container_width=True):
                    if (i+1) in selected:
                        selected.remove(i+1)
                    else:
                        selected.add(i+1)
                st.image(thumb, use_column_width=True)
                chk = st.checkbox(f"Выбрать {i+1}", key=f"chk_{i}", value=(i+1) in selected)
                if chk:
                    selected.add(i+1)
                else:
                    selected.discard(i+1)
        st.session_state["selected_pages"] = selected
    with col2:
        preset = st.text_input("Диапазоны страниц (пример: 1,3,5-7)")
        if st.button("Добавить из диапазона"):
            pages_from_range = parse_page_ranges(preset, total)
            sel = st.session_state.get("selected_pages", set())
            sel.update(pages_from_range)
            st.session_state["selected_pages"] = sel
        if st.button("Очистить выбор"):
            st.session_state["selected_pages"] = set()
        st.write("Текущий выбор:")
        sel_sorted = sorted(st.session_state.get("selected_pages", set()))
        st.code(", ".join(map(str, sel_sorted)) or "—")

sel_pages = sorted(st.session_state.get("selected_pages", set()))
if not sel_pages:
    st.warning("Не выбраны страницы. Выберите хотя бы одну.")
    st.stop()

st.subheader("Предпросмотр выбранных страниц")
cols = st.columns(4)
for j, p in enumerate(sel_pages[:8]):
    with cols[j % 4]:
        st.image(render_page_thumbnail(pdf_bytes, p-1), caption=f"Стр. {p}", use_column_width=True)

st.subheader("Извлечение текста (по текущей активной странице)")
active_page = st.selectbox("Страница для быстрого текста", sel_pages, index=0)
text_preview = extract_text_from_page(pdf_bytes, active_page-1)
st.text_area("Текст", text_preview[:4000], height=240)

st.subheader("Таблицы и колонка ‘Наименование’")
df, raw = extract_tables_with_name(pdf_bytes, sel_pages)

if df.empty:
    st.info("Таблицы с колонкой ‘Наименование’ не найдены на выбранных страницах. Попробуйте другие страницы или используйте OCR/другие параметры в парсере.")
else:
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("⬇️ Скачать CSV", data=csv, file_name="extracted_naimenovanie.csv", mime="text/csv")

st.divider()
st.caption("Примечание: OCR для сканов не включён. Для сканов добавьте Tesseract/Cloud Vision и подмените извлечение текста/таблиц.")
