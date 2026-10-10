from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthProfile(BaseModel):
    """
    Generic health profile used by models that require
    a general health input structure.
    """

    age: int = Field(..., ge=18, le=120)
    sex: Optional[str] = None
    bmi: Optional[float] = Field(default=None, ge=10, le=80)
    systolic_bp: Optional[float] = Field(default=None, ge=60, le=260)
    diastolic_bp: Optional[float] = Field(default=None, ge=30, le=180)
    glucose: Optional[float] = Field(default=None, ge=30, le=700)
    cholesterol: Optional[float] = Field(default=None, ge=50, le=600)
    smoking: Optional[bool] = None
    physical_activity: Optional[str] = None
    family_history: Optional[bool] = None


class DiabetesProfile(BaseModel):
    """
    Input schema matching the 21 features used by the
    UCI CDC Diabetes Health Indicators model.

    Most fields use the original dataset's 0/1 or ordinal encoding.
    """

    high_bp: int = Field(..., ge=0, le=1)
    high_chol: int = Field(..., ge=0, le=1)
    chol_check: int = Field(..., ge=0, le=1)

    bmi: float = Field(..., ge=10, le=98)

    smoker: int = Field(..., ge=0, le=1)
    stroke: int = Field(..., ge=0, le=1)
    heart_disease_or_attack: int = Field(..., ge=0, le=1)

    physical_activity: int = Field(..., ge=0, le=1)
    fruits: int = Field(..., ge=0, le=1)
    veggies: int = Field(..., ge=0, le=1)
    heavy_alcohol_consumption: int = Field(..., ge=0, le=1)

    any_healthcare: int = Field(..., ge=0, le=1)
    no_doc_bc_cost: int = Field(..., ge=0, le=1)

    general_health: int = Field(..., ge=1, le=5)
    mental_health_days: int = Field(..., ge=0, le=30)
    physical_health_days: int = Field(..., ge=0, le=30)

    difficulty_walking: int = Field(..., ge=0, le=1)

    sex: int = Field(..., ge=0, le=1)

    age_category: int = Field(..., ge=1, le=13)
    education: int = Field(..., ge=1, le=6)
    income: int = Field(..., ge=1, le=8)


class CardiovascularProfile(BaseModel):
    """
    Input schema matching the 13 features used by the
    UCI Heart Disease Logistic Regression model.
    """

    age: int = Field(..., ge=18, le=100)

    # Dataset encoding: 0 = female, 1 = male
    sex: int = Field(..., ge=0, le=1)

    # Original dataset categories
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=60, le=260)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)

    # Optional because the original dataset has missing values
    ca: Optional[float] = Field(default=None, ge=0, le=3)
    thal: Optional[float] = Field(default=None, ge=3, le=7)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    disease: str
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    model_version: str
    explanation: list[str]
    disclaimer: str