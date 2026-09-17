#!/usr/bin/env python3
"""
Render a markdown onboarding document into a Word (.docx) file.

Usage: python md_to_docx.py <input.md> [output.docx]
"""
import sys
import os
import markdown
from htmldocx import HtmlToDocx
from docx import Document


def convert(md_path, docx_path=None):
    if docx_path is None:
        docx_path = os.path.splitext(md_path)[0] + ".docx"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = markdown.markdown(md_text, extensions=["extra", "sane_lists", "nl2br"])

    document = Document()
    parser = HtmlToDocx()
    parser.add_html_to_document(html, document)
    document.save(docx_path)

    print(f"[md_to_docx] {md_path} -> {docx_path}")
    return docx_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_docx.py <input.md> [output.docx]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
