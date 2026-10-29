import json
import re
import textwrap
from typing import Callable, Dict, Union

Content = Union[str, Dict]


def _to_markdown(content: Content) -> str:
    if isinstance(content, str):
        return content
    lines = []
    for key in ("objects", "goals", "activities"):
        if key in content:
            lines.append(f"## {key.capitalize()}")
            values = content[key] if isinstance(content[key], list) else [content[key]]
            lines += [f"- {v}" for v in values] + [""]
    return "\n".join(lines) or json.dumps(content, ensure_ascii=False, indent=2)


def _sections(markdown: str):
    sections, current = [], {"heading": None, "level": 0, "lines": []}
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if current["heading"] is not None or any(l.strip() for l in current["lines"]):
                sections.append(current)
            current = {"heading": m.group(2).strip(), "level": len(m.group(1)), "lines": []}
        else:
            current["lines"].append(line)
    sections.append(current)
    return [{"heading": s["heading"], "level": s["level"], "content": "\n".join(s["lines"]).strip()} for s in sections]


def render_markdown(content: Content) -> bytes:
    return _to_markdown(content).encode("utf-8")


def render_json(content: Content) -> bytes:
    data = content if isinstance(content, dict) else {"sections": _sections(content)}
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def render_html(content: Content) -> bytes:
    markdown_text = _to_markdown(content)
    try:
        import markdown
        body = markdown.markdown(markdown_text, extensions=["extra", "sane_lists"])
    except ImportError:
        body = "<pre>" + markdown_text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>{body}</body></html>".encode("utf-8")


def _plain_lines(markdown_text, width=95):
    out = []
    for raw in markdown_text.splitlines():
        line = re.sub(r"<sub>(.*?)</sub>", r"\1", raw)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"[*`]", "", line)
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            out.append(("H", heading.group(2)))
            continue
        if not line.strip():
            out.append(("T", ""))
            continue
        indent = len(line) - len(line.lstrip())
        for part in textwrap.wrap(line.strip(), width=width - indent) or [""]:
            out.append(("T", " " * indent + part))
    return out


def _pdf_escape(text):
    return text.encode("latin-1", "replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(content: Content) -> bytes:
    lines = _plain_lines(_to_markdown(content))
    page_h, top, bottom, left, step = 842, 792, 50, 50, 14
    pages, current, y = [], [], top
    for kind, text in lines:
        if y < bottom:
            pages.append(current)
            current, y = [], top
        font, size = ("F2", 12) if kind == "H" else ("F1", 10)
        current.append(f"BT /{font} {size} Tf {left} {y} Td ({_pdf_escape(text)}) Tj ET")
        y -= step + (4 if kind == "H" else 0)
    pages.append(current)
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    kids = []
    for ops in pages:
        stream = "\n".join(ops).encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 {page_h}] "
                       f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, obj in enumerate(objects, 1):
        offsets.append(len(out))
        body = obj if isinstance(obj, bytes) else obj.encode("latin-1")
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


RENDERERS: Dict[str, Callable[[Content], bytes]] = {
    "markdown": render_markdown,
    "json": render_json,
    "html": render_html,
    "pdf": render_pdf,
}

EXTENSIONS = {"markdown": "md", "json": "json", "html": "html", "pdf": "pdf"}
MEDIA_TYPES = {"markdown": "text/markdown; charset=utf-8", "json": "application/json",
               "html": "text/html; charset=utf-8", "pdf": "application/pdf"}


def register_renderer(name, fn, extension=None, media_type="application/octet-stream"):
    RENDERERS[name] = fn
    EXTENSIONS[name] = extension or name
    MEDIA_TYPES[name] = media_type


def render(content: Content, fmt="markdown") -> bytes:
    if fmt not in RENDERERS:
        raise ValueError(f"unknown format: {fmt}")
    return RENDERERS[fmt](content)


def write(content: Content, path, fmt=None):
    fmt = fmt or next((k for k, v in EXTENSIONS.items() if str(path).endswith("." + v)), "markdown")
    data = render(content, fmt)
    with open(path, "wb") as f:
        f.write(data)
    return path
