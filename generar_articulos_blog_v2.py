#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_articulos_blog_v2.py

Genera artículos para el blog a partir de Markdown en ./documentos,
usando Ollama local y guardándolos en SQLite.

Características:
- Cada heading ### genera 1 artículo.
- category = "<archivo>, <H1 actual>, <H2 actual>"
- Dedupe por (title, category)
- Cache en .cache_articulos/
- Soporta Ollama /api/chat y /api/generate
- Quality gate + auto-repair (1 reintento si sale corto o faltan secciones)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import requests


# =========================
# CONFIG
# =========================
MODEL_DEFAULT = "llama3:latest"
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434/api/chat"  # si tu Ollama no tiene chat, usa /api/generate

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR / "documentos"
DB_PATH = SCRIPT_DIR / "blog.sqlite"

REQUEST_TIMEOUT = 240
RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.6

CACHE_DIR = SCRIPT_DIR / ".cache_articulos"
CACHE_DIR.mkdir(exist_ok=True)


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class H3Item:
    file_path: Path
    file_stem: str
    h1: str
    h2: str
    h3_raw: str
    h3_title: str
    category: str


# =========================
# HELPERS
# =========================
def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT NOT NULL,
              title TEXT NOT NULL,
              content TEXT NOT NULL,
              category TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(date DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_category ON posts(category);")
        # evita duplicados si dos procesos corren a la vez
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_posts_title_category ON posts(title, category);"
        )
        conn.commit()
    finally:
        conn.close()


def post_exists(conn: sqlite3.Connection, title: str, category: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM posts WHERE title = ? AND category = ? LIMIT 1",
        (title, category),
    )
    return cur.fetchone() is not None


def insert_post(conn: sqlite3.Connection, title: str, content_md: str, category: str) -> None:
    conn.execute(
        "INSERT INTO posts(date, title, content, category) VALUES(?, ?, ?, ?)",
        (now_iso(), title, content_md, category),
    )
    conn.commit()


def strip_numbering_from_h3(title: str) -> str:
    """
    "Lección 1.1.1 — Por qué ajustar pesos es difícil" -> "Por qué ajustar pesos es difícil"
    """
    t = title.strip()
    t = re.sub(r"^(lecci[oó]n|lesson|tema|cap[ií]tulo|unidad)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*\d+(?:[\.\-]\d+){0,6}\s*[\)\.\-–—:]*\s*", "", t)
    t = re.sub(r"^\s*[–—-]\s*", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t if t else title.strip()


def extract_front_matter(md: str) -> Tuple[Dict[str, str], str]:
    md = normalize_newlines(md)
    if not md.startswith("---\n"):
        return {}, md

    parts = md.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, md

    fm_block = parts[0].splitlines()[1:]
    body = parts[1]

    fm: Dict[str, str] = {}
    for line in fm_block:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)\s*$", line)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip().strip('"').strip("'")
            fm[k] = v

    return fm, body


def extract_h3_items(md_body: str, file_path: Path) -> List[H3Item]:
    """
    Recorre el documento en orden y para cada ### captura el H1 y H2 más recientes.
    """
    h1_current = ""
    h2_current = ""
    items: List[H3Item] = []

    file_stem = file_path.stem.strip() or file_path.name

    for line in normalize_newlines(md_body).splitlines():
        m1 = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if m1:
            h1_current = m1.group(1).strip()
            h2_current = ""
            continue

        m2 = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m2:
            h2_current = m2.group(1).strip()
            continue

        m3 = re.match(r"^\s*###\s+(.+?)\s*$", line)
        if m3:
            raw_h3 = m3.group(1).strip()
            title = strip_numbering_from_h3(raw_h3)

            h1 = h1_current.strip() if h1_current.strip() else "Sin sección principal"
            h2 = h2_current.strip() if h2_current.strip() else "Sin subsección"
            category = f"{file_stem}, {h1}, {h2}"

            items.append(
                H3Item(
                    file_path=file_path,
                    file_stem=file_stem,
                    h1=h1,
                    h2=h2,
                    h3_raw=raw_h3,
                    h3_title=title,
                    category=category,
                )
            )

    return items


def extract_section_context(md_body: str, h1: str, h2: str, h3_raw: str, max_chars: int = 12000) -> str:
    """
    Contexto centrado en el H3:
    - Encabezados (#, ##, ###) para situar el tema
    - Texto desde ese ### hasta antes del siguiente ### o un ##/#
    """
    lines = normalize_newlines(md_body).splitlines()
    out: List[str] = []
    in_block = False
    found = False

    for line in lines:
        m3 = re.match(r"^\s*###\s+(.+?)\s*$", line)
        if m3:
            current = m3.group(1).strip()
            if current == h3_raw:
                in_block = True
                found = True
                continue
            elif in_block:
                break

        if in_block and re.match(r"^\s*#{1,2}\s+", line):
            break

        if in_block:
            out.append(line)

    header = f"# {h1}\n## {h2}\n### {h3_raw}\n"
    chunk = (header + "\n".join(out)).strip()

    if not found:
        chunk = header.strip()

    if len(chunk) > max_chars:
        chunk = chunk[:max_chars].rstrip() + "\n\n*(Contexto truncado por límite de tamaño)*\n"

    return chunk


def build_prompt(section_context: str, category: str, article_title: str) -> Tuple[str, str]:
    system = (
        "Eres un redactor técnico senior especializado en IA aplicada a programación. "
        "Escribes en español, con un tono claro, práctico y profesional para programadores. "
        "No inventas enlaces, fuentes ni datos. No añades relleno innecesario."
    )

    user = f"""CONTEXTO (sección relevante del temario):
\"\"\"{section_context}\"\"\"

METADATOS:
- Categoría: {category}
- Título: "{article_title}"

TAREA:
Genera un artículo en Markdown. Longitud objetivo: 900–1400 palabras.

FORMATO DE SALIDA (estricto):
1) Primera línea: # {article_title}
2) Segunda línea: **Meta:** <meta descripción 140–160 caracteres>
3) Luego el contenido.

Estructura mínima:
- Introducción (por qué importa)
- Explicación principal con ejemplos (incluye 1 bloque de código corto si ayuda)
- Errores típicos / trampas (>=3)
- Checklist accionable (5–10 puntos)
- Siguientes pasos (2–4 bullets)

REGLAS:
- NO incluyas YAML front-matter.
- NO incluyas enlaces inventados.
- NO incluyas frases tipo “Aquí tienes…”.
- Devuelve SOLO Markdown.
"""
    return system, user


def repair_prompt(original_md: str, title: str, reason: str) -> Tuple[str, str]:
    system = (
        "Eres un editor técnico senior. Mejoras artículos Markdown en español. "
        "No inventas enlaces ni metes paja. Mantienes coherencia, y amplías si hace falta."
    )
    user = f"""El artículo siguiente ha fallado el control de calidad por: {reason}

OBJETIVO (estricto):
- Mantén el mismo tema y el mismo título.
- Asegura que el artículo tenga entre 900–1400 palabras (si está corto, amplía con contenido útil).
- Debe incluir: **Meta:**, sección de Errores, Checklist, y Siguientes pasos.
- Añade ejemplos concretos y explicación más profunda (sin inventar fuentes/enlaces).
- Devuelve SOLO Markdown.

ARTÍCULO ACTUAL:
\"\"\"{original_md}\"\"\"
"""
    return system, user


def ollama_call(model: str, url: str, system: str, user: str) -> str:
    """
    Soporta /api/chat y /api/generate según la URL.
    - chat: devuelve {"message":{"content":"..."}}
    - generate: devuelve {"response":"..."}
    """
    last_err = None

    for attempt in range(1, RETRIES + 1):
        try:
            if url.endswith("/api/chat"):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
                }
                r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                text = ((data.get("message") or {}).get("content") or "").strip()
            else:
                payload = {
                    "model": model,
                    "prompt": f"{system}\n\n{user}",
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
                }
                r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                text = (data.get("response") or "").strip()

            if not text:
                raise RuntimeError("Respuesta vacía de Ollama.")
            return text

        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Fallo llamando a Ollama tras {RETRIES} intentos: {last_err}")


def approx_word_count(md: str) -> int:
    return len(re.findall(r"\w+", md))


def quality_check(md: str, min_words: int) -> Tuple[bool, str]:
    text = (md or "").strip()

    if not text.startswith("# "):
        return False, "No empieza con H1 (# ...)"

    wc = approx_word_count(text)
    if wc < min_words:
        return False, f"Muy corto (~{wc} palabras)"

    needed = ["**Meta:**", "Errores", "Checklist", "Siguientes pasos"]
    missing = [k for k in needed if k.lower() not in text.lower()]
    if missing:
        return False, f"Faltan secciones: {missing}"

    # Anti-repetición básica
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        top = max((lines.count(ln) for ln in set(lines)), default=1)
        if top >= 6:
            return False, "Repetición excesiva de líneas"

    return True, "OK"


def cache_path(file_path: Path, title: str, category: str, model: str) -> Path:
    key = f"{file_path.name}||{title}||{category}||{model}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


# =========================
# MAIN
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--ollama-url", default=OLLAMA_URL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="No escribe en DB (solo simula).")
    ap.add_argument("--limit", type=int, default=0, help="Máximo de artículos a generar (0 = sin límite).")
    ap.add_argument("--min-words", type=int, default=200, help="Mínimo de palabras aproximadas para pasar QA.")
    ap.add_argument("--repair", action="store_true", default=True, help="Reintenta 1 vez si falla QA por corto/secciones.")
    args = ap.parse_args()

    if not DOCS_DIR.is_dir():
        raise SystemExit(f"ERROR: No existe la carpeta: {DOCS_DIR}")

    ensure_db(DB_PATH)

    md_files = sorted([p for p in DOCS_DIR.glob("*.md") if p.is_file()])
    if not md_files:
        print(f"No se encontraron .md en {DOCS_DIR}")
        return

    print(f"Encontrados {len(md_files)} archivos en {DOCS_DIR}")
    print(f"DB: {DB_PATH}  | dry-run={args.dry_run}")
    print(f"Modelo: {args.model}")
    print(f"Ollama: {args.ollama_url}")
    print(f"QA: min_words={args.min_words} | repair={args.repair}")
    print("-" * 70)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        inserted = 0
        skipped = 0
        rejected = 0
        generated = 0

        for md_path in md_files:
            raw = read_text(md_path)
            _fm, body = extract_front_matter(raw)

            items = extract_h3_items(body, file_path=md_path)
            if not items:
                print(f"[SKIP] {md_path.name}: no hay headings ###")
                continue

            print(f"\n[{md_path.name}] H3 encontrados: {len(items)}")

            for it in items:
                if args.limit and generated >= args.limit:
                    print("\n[STOP] Alcanzado --limit")
                    break

                title = it.h3_title
                category = it.category

                if post_exists(conn, title, category):
                    skipped += 1
                    print(f"  - (skip) Ya existe: [{category}] {title}")
                    continue

                ck = cache_path(md_path, title, category, args.model)
                if ck.is_file():
                    try:
                        cached = json.loads(ck.read_text(encoding="utf-8"))
                        content_md = (cached.get("content") or "").strip()
                        ok, reason = quality_check(content_md, args.min_words) if content_md else (False, "Cache vacío")
                        if ok:
                            if not args.dry_run:
                                insert_post(conn, title, content_md, category)
                            inserted += 1
                            print(f"  - (cache→db) [{category}] {title}")
                            continue
                        else:
                            print(f"  - (cache reject) {title}: {reason} (regenerando)")
                    except Exception:
                        pass

                # Contexto centrado
                section_ctx = extract_section_context(body, it.h1, it.h2, it.h3_raw)
                system, user = build_prompt(section_ctx, category, title)

                print(f"  - (gen) [{category}] {title}")
                content_md = ollama_call(args.model, args.ollama_url, system, user)
                generated += 1

                ok, reason = quality_check(content_md, args.min_words)

                # Auto-repair si procede
                if (not ok) and args.repair and ("Muy corto" in reason or "Faltan secciones" in reason):
                    system2, user2 = repair_prompt(content_md, title, reason)
                    print(f"  - (repair) {title}: {reason}")
                    content_md2 = ollama_call(args.model, args.ollama_url, system2, user2)
                    ok2, reason2 = quality_check(content_md2, args.min_words)
                    if ok2:
                        content_md = content_md2
                        ok, reason = ok2, reason2
                    else:
                        reason = f"{reason} | Repair falló: {reason2}"

                meta = {
                    "source_file": md_path.name,
                    "category": category,
                    "title": title,
                    "generated_at": now_iso(),
                    "model": args.model,
                    "ok": ok,
                    "reason": reason,
                    "hierarchy": {"h1": it.h1, "h2": it.h2, "h3_raw": it.h3_raw},
                    "ollama_url": args.ollama_url,
                }

                ck.write_text(
                    json.dumps({**meta, "content": content_md}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                if not ok:
                    rejected += 1
                    print(f"  - (reject) {title}: {reason}")
                    continue

                if not args.dry_run:
                    insert_post(conn, title, content_md, category)

                inserted += 1
                time.sleep(SLEEP_BETWEEN_CALLS)

            if args.limit and generated >= args.limit:
                break

        print("\n" + "=" * 70)
        print(f"Generados (llamadas a IA): {generated}")
        print(f"Insertados: {inserted}")
        print(f"Saltados (ya existían): {skipped}")
        print(f"Rechazados (QA): {rejected}")
        print(f"Cache: {CACHE_DIR}")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
