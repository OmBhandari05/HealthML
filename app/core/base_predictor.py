from abc import ABC, abstractmethod
from app.schemas.health import HealthProfile, PredictionResponse

class BasePredictor(ABC):
    name: str
    version: str

    @abstractmethod
    def predict(self, profile: HealthProfile) -> PredictionResponse:
        raise NotImplementedError
