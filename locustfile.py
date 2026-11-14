from locust import HttpUser, task, between
import uuid


class PreauthUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def submit_preauthorisation(self):
        payload = {
            "request_id": f"LOAD-{uuid.uuid4().hex[:10]}",
            "policy_id": "POL-1001",
            "policy_start_date": "2025-01-01",
            "admission_date": "2026-11-20",
            "admission_type": "planned_surgical",
            "procedure_name": "Cataract Surgery",
            "requested_amount": 30000,
            "policy_limit": 500000,
            "required_documents_complete": True,
            "room_category": "Single"
        }

        with self.client.post(
            "/preauth/async",
            json=payload,
            name="/preauth/async",
            catch_response=True
        ) as response:
            if 200 <= response.status_code < 300:
                try:
                    data = response.json()
                    if data.get("run_id") and data.get("status") == "queued":
                        response.success()
                    else:
                        response.failure("Response missing run_id or queued status")
                except ValueError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
