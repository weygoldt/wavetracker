"""Inject the real-data crops into the explainer template."""
import json
from pathlib import Path
here = Path(__file__).parent
crop = json.loads((here / "cropB.json").read_text())
elec = json.loads((here / "elecB.json").read_text())
html = (here / "template.html").read_text()
html = html.replace("/*CROP_JSON*/", json.dumps(crop, separators=(",", ":")))
html = html.replace("/*ELEC_JSON*/", json.dumps(elec, separators=(",", ":")))
assert "</script" not in json.dumps(crop) + json.dumps(elec)
(here / "learned-detector.html").write_text(html)
print(len(html) // 1024, "KB")
