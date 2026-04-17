from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.json_io import write_json_atomic


class JsonIoTests(unittest.TestCase):
    def test_write_json_atomic_creates_expected_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "summary.json"

            write_json_atomic(out_path, {"ok": True, "value": 3})

            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), {"ok": True, "value": 3})

    def test_write_json_atomic_replaces_existing_file_without_temp_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_path = tmp_path / "summary.json"
            out_path.write_text('{"stale": true}\n', encoding="utf-8")

            write_json_atomic(out_path, {"fresh": True})

            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), {"fresh": True})
            self.assertEqual(list(tmp_path.glob("summary.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
