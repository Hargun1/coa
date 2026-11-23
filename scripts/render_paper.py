"""Render each PDF page and flag text outside page boundaries for visual QA."""

import json
from pathlib import Path
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]


def render():
    directory = ROOT / "tmp/paper"
    directory.mkdir(parents=True, exist_ok=True)
    for old_page in directory.glob("page-*.png"):
        old_page.unlink()
    doc = fitz.open(ROOT / "output/pdf/research_paper.pdf")
    failures = []
    for index, page in enumerate(doc):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < 20 or y0 < 15 or x1 > page.rect.width - 20 or y1 > page.rect.height - 15:
                failures.append({"page": index + 1, "box": [x0, y0, x1, y1], "text": block[4]})
        page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(directory / f"page-{index+1:02}.png")
    text = "\n".join(page.get_text() for page in doc)
    assert len(text) > 15000
    assert "References" in text and "Conclusion" in text
    assert not failures, failures
    result = {"pages": len(doc), "text_boundary_checks": "passed", "visual_review": "required separately"}
    (directory / "render.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    render()

