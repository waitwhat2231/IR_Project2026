"""Convert the HTML report to a clean PDF (no browser header/footer) via Chrome DevTools Protocol."""
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
HERE = Path(__file__).parent
HTML = HERE / "IR_Project_Report.html"
PDF = HERE / "IR_Project_Report.pdf"
PORT = 9222


def main() -> int:
    proc = subprocess.Popen([
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--remote-debugging-port={PORT}",
        "--remote-allow-origins=*",
        "--user-data-dir=" + str(Path.home() / "chrome_cdp_pdf"),
        "about:blank",
    ])
    try:
        # Wait for the DevTools endpoint to come up
        ws_url = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version") as r:
                    ws_url = json.load(r)["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.5)
        if not ws_url:
            print("Could not reach Chrome DevTools endpoint")
            return 1

        # Use the existing page target (Chrome was launched with about:blank)
        page_ws = None
        for _ in range(20):
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list") as r:
                targets = json.load(r)
            pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            if pages:
                page_ws = pages[0]["webSocketDebuggerUrl"]
                break
            time.sleep(0.5)
        if not page_ws:
            print("No page target found")
            return 1

        ws = websocket.create_connection(page_ws, max_size=200_000_000)
        mid = 0

        def cmd(method, params=None):
            nonlocal mid
            mid += 1
            ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == mid:
                    return msg

        cmd("Page.enable")
        file_url = HTML.resolve().as_uri()
        cmd("Page.navigate", {"url": file_url})

        # Wait for the page load event
        deadline = time.time() + 30
        while time.time() < deadline:
            msg = json.loads(ws.recv())
            if msg.get("method") == "Page.loadEventFired":
                break
        time.sleep(1.5)  # let fonts/layout settle

        result = cmd("Page.printToPDF", {
            "printBackground": True,
            "displayHeaderFooter": False,
            "preferCSSPageSize": True,
        })
        data = result["result"]["data"]
        PDF.write_bytes(base64.b64decode(data))
        ws.close()
        print(f"PDF OK size={PDF.stat().st_size}")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
