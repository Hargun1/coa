"""Optional browser acceptance check; requires Playwright and Chromium.

Run after starting the local server. Screenshots go under tmp/browser.
Playwright is a verification dependency, not needed to run the project.
"""

import json
import os
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def check():
    directory = ROOT / "tmp/browser"
    directory.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=os.environ.get("LAB_BROWSER_EXECUTABLE"))
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        expect(page.locator("#status")).to_contain_text("PAPER RESULTS", timeout=15000)
        assert page.locator("#policy-table tr").count() == 7
        assert page.locator("#timeline polyline").count() == 3
        assert page.locator("#amat").inner_text() != "—"
        page.locator("#workload").select_option("mixed_scan")
        page.locator("#capacity").select_option("256")
        saved_amat = page.locator("#amat").inner_text()
        page.screenshot(path=str(directory / "desktop.png"), full_page=True)
        page.locator("#seed").fill("23")
        page.locator("#run").click()
        expect(page.locator("#status")).to_contain_text("INTERACTIVE DEMO", timeout=60000)
        assert "1 seed" in page.locator("#status").inner_text()
        assert page.locator("#policy-table tr").count() == 7
        assert not page.locator("#run").is_disabled()
        page.locator("#restore").click()
        assert page.locator("#amat").inner_text() == saved_amat
        pdf = page.request.get("http://127.0.0.1:8765/paper.pdf")
        assert pdf.status == 200 and pdf.body().startswith(b"%PDF")
        # Exercise the same comparison on a phone-sized viewport.
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(directory / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"
        assert page.locator("#run").is_visible()
        browser.close()
    assert not errors, errors
    result = {"status": "passed", "console_errors": errors, "checked": ["saved results", "workload/capacity controls", "live seven-policy run", "restore", "paper PDF", "390px responsive layout"]}
    (directory / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    check()

