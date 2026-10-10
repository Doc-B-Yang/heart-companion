"""Exercise browser preflights without hardware, personal data or LLM calls."""
import unittest
from fastapi.testclient import TestClient
from hrv_companion_agent.backend.main import app


class TestBrowserPreflight(unittest.TestCase):
    def test_published_page_get_and_sensor_post(self):
        with TestClient(app) as client:
            for path, method in (("/api/emotion/state?user_id=user001", "GET"),
                                 ("/api/sensor/raw", "POST")):
                response = client.options(path, headers={
                    "Origin": "https://doc-b-yang.github.io",
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["access-control-allow-origin"],
                                 "https://doc-b-yang.github.io")

    def test_unrelated_website_rejected(self):
        with TestClient(app) as client:
            response = client.options("/api/sensor/raw", headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })
            self.assertEqual(response.status_code, 400)
            self.assertNotIn("access-control-allow-origin", response.headers)
