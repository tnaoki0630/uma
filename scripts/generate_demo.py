import json
from pathlib import Path

from uma_api.demo_data import build_demo_catalog
from uma_api.schemas import PredictionRequest
from uma_api.service import predict

root = Path(__file__).parent.parent
payload = json.loads((root / "web" / "demo-race.json").read_text())
result = predict(PredictionRequest.model_validate(payload))


def stable_numbers(value):
    """Make the checked-in demo stable across BLAS implementations."""
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, list):
        return [stable_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: stable_numbers(item) for key, item in value.items()}
    return value


(root / "web" / "demo-result.json").write_text(
    json.dumps(stable_numbers(result.model_dump(mode="json")), ensure_ascii=False, indent=2) + "\n"
)
(root / "web" / "demo-catalog.json").write_text(
    json.dumps(build_demo_catalog(root / "data"), ensure_ascii=False, indent=2) + "\n"
)
