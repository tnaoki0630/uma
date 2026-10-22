import json
from pathlib import Path

from uma_api.schemas import PredictionRequest
from uma_api.service import predict

root = Path(__file__).parent.parent
payload = json.loads((root / "web" / "demo-race.json").read_text())
result = predict(PredictionRequest.model_validate(payload))
(root / "web" / "demo-result.json").write_text(
    json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
)
