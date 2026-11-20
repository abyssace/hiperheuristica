"""Convierte report/REPORTE.md a report/REPORTE.pdf (el PDF no se versiona).

Uso:
    python -m src.build_report
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FONTS = Path("/System/Library/Fonts/Supplemental")  # macOS; cambia la ruta en otro sistema


def main() -> None:
    text = (REPORT / "REPORTE.md").read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    # Rutas de imagen absolutas y ancho fijo; cada figura en su propio bloque.
    html = re.sub(r'<p><img alt="[^"]*" src="\.\./([^"]+)" /></p>',
                  lambda m: f'<center><img src="{ROOT / m.group(1)}" width="480"></center>', html)
    html = html.replace("<table>", '<table border="1" cellpadding="3" width="100%">')
    # fpdf2 no admite etiquetas anidadas dentro de celdas: se deja solo el texto.
    html = re.sub(r"</?(thead|tbody)>", "", html)
    html = re.sub(r"<(t[dh])([^>]*)>(.*?)</\1>",
                  lambda m: f"<{m.group(1)}{m.group(2)}>{re.sub('<[^>]+>', '', m.group(3))}</{m.group(1)}>",
                  html, flags=re.S)
    pdf = FPDF(format="Letter")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    for style, name in (("", "Arial"), ("B", "Arial Bold"), ("I", "Arial Italic"), ("BI", "Arial Bold Italic")):
        pdf.add_font("Arial", style, str(FONTS / f"{name}.ttf"))
    for style, name in (("", "Courier New"), ("B", "Courier New Bold"), ("I", "Courier New Italic"),
                        ("BI", "Courier New Bold Italic")):
        pdf.add_font("Mono", style, str(FONTS / f"{name}.ttf"))
    pdf.set_font("Arial", size=10)
    pdf.add_page()
    ink = (11, 11, 11)
    styles = {f"h{level}": FontFace(family="Arial", emphasis="B", color=ink, size_pt=size)
              for level, size in ((1, 19), (2, 14), (3, 11.5))}
    styles["code"] = FontFace(family="Mono", size_pt=9)
    styles["pre"] = FontFace(family="Mono", size_pt=9)
    pdf.write_html(html, font_family="Arial", tag_styles=styles, table_line_separators=True)
    pdf.output(str(REPORT / "REPORTE.pdf"))
    print(f"Escrito {REPORT / 'REPORTE.pdf'} ({pdf.page_no()} páginas)")


if __name__ == "__main__":
    main()
