HealthML — Healthcare Risk Assessment Platform
HealthML is a modular machine-learning platform that uses trained
classification models to generate educational risk assessments for
diabetes and cardiovascular disease. It combines a React frontend,
FastAPI backend, and independently trained ML pipelines.
Disclaimer: HealthML is an educational and research prototype. It
is not a medical diagnosis system, and its model scores are not
clinically calibrated probabilities. Do not use its predictions as a
substitute for professional medical advice.

Features
- Diabetes risk assessment using a HistGradientBoosting classifier.
- Cardiovascular disease assessment using Logistic Regression.
- Interactive React frontend with separate assessment forms and result
  pages.
- FastAPI REST API with input validation using Pydantic.
- Model score, decision threshold, model version, and input observations
  displayed in results.
- Automated API tests using pytest.
- Modular architecture designed to support additional disease models in
  the future.
Technology Stack
Component	Technology
Frontend	React, Vite, JavaScript, CSS
Backend	Python, FastAPI, Uvicorn
Machine learning	scikit-learn
Data processing	Pandas, NumPy
Model persistence	Joblib
Validation	Pydantic
Secure dataset downloads	truststore
Testing	Pytest, HTTPX


Project Architecture
HealthML/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── base_predictor.py
│   │   └── registry.py
│   ├── models/
│   │   ├── diabetes/
│   │   │   ├── model.joblib
│   │   │   ├── metadata.json
│   │   │   └── predictor.py
│   │   └── cardiovascular/
│   │       ├── model.joblib
│   │       ├── metadata.json
│   │       └── predictor.py
│   ├── schemas/
│   │   └── health.py
│   └── main.py
├── data/
│   ├── download_diabetes.py
│   ├── raw/                 # Downloaded datasets (not committed)
│   ├── processed/           # Generated train/test data (not committed)
│   └── scripts/
│       ├── audit_diabetes.py
│       ├── eda_diabetes.py
│       ├── prepare_diabetes.py
│       ├── train_diabetes_models.py
│       ├── tune_diabetes_model.py
│       ├── tune_hgb.py
│       ├── threshold_analysis.py
│       ├── final_diabetes_evaluation.py
│       ├── save_diabetes_model.py
│       ├── eda_cardiovascular.py
│       ├── prepare_cardiovascular.py
│       ├── train_cardiovascular_models.py
│       ├── tune_cardiovascular_model.py
│       ├── final_cardiovascular_evaluation.py
│       └── save_cardiovascular_model.py
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── reports/
│   └── figures/
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
The structure above highlights the main project components. Some
directories contain additional scripts and files.
Datasets
1. Diabetes Health Indicators
- Source: UCI Machine Learning Repository — CDC Diabetes Health
  Indicators
- Records: 253,680
- Original columns: 23, including an identifier, 21 input features,
  and the target.
- Model features: 21
- Target: Diabetes_binary
The target represents whether a respondent belongs to the dataset's
diabetes/prediabetes-positive group. The identifier column is excluded
from model training.
2. Cardiovascular Disease
- Source: UCI Machine Learning Repository — Heart
  Disease
- Records: 303
- Original input features: 13
- Original target: num
For binary classification, the target is transformed so that num = 0
represents no disease and num > 0 represents disease present.
Important limitation: The cardiovascular dataset is small. Its
evaluation metrics may vary substantially with different data splits and
should not be interpreted as evidence of clinical readiness.
Machine Learning Workflow
The project follows a structured workflow:
1. Dataset download and audit.
2. Exploratory data analysis and target distribution checks.
3. Feature preparation and target separation.
4. Stratified train/test splitting.
5. Model training and validation.
6. Hyperparameter and decision-threshold selection.
7. Final evaluation on a held-out test set.
8. Model persistence and integration into the API.
Preprocessing is fitted on training data rather than the held-out test
data. The final test set is reserved for final evaluation.
Reproducing the ML Pipeline
Run these commands from the project root after installing the Python
dependencies.
Diabetes model
The downloader uses HTTPS certificate verification and the truststore
dependency.
python data/download_diabetes.py
python data/scripts/audit_diabetes.py
python data/scripts/eda_diabetes.py
python data/scripts/prepare_diabetes.py
python data/scripts/train_diabetes_models.py
python data/scripts/tune_diabetes_model.py
python data/scripts/tune_hgb.py
python data/scripts/threshold_analysis.py
python data/scripts/final_diabetes_evaluation.py
python data/scripts/save_diabetes_model.py
Cardiovascular model
First, obtain the Heart Disease dataset from the UCI Machine Learning
Repository and
place the dataset at data/raw/heart_disease.csv. Then run:
python data/scripts/eda_cardiovascular.py
python data/scripts/prepare_cardiovascular.py
python data/scripts/train_cardiovascular_models.py
python data/scripts/tune_cardiovascular_model.py
python data/scripts/final_cardiovascular_evaluation.py
python data/scripts/save_cardiovascular_model.py
These commands regenerate processed data, evaluation outputs, and saved
model artifacts. The saved models are already included in the
repository, so retraining is not required just to run the application.
Metrics may vary if datasets, software versions, random seeds,
preprocessing, or training configurations change. Evaluate retrained
models before integrating them into the application.
Selected Models
Disease module	Selected model	Decision threshold
Diabetes	HistGradientBoosting	0.65
Cardiovascular	Logistic Regression	0.60


The thresholds were selected during model validation. They are model
decision thresholds, not medical cutoffs.
Model Evaluation
Diabetes
Final held-out test results:
Metric	Score
Accuracy	80.41%
Precision	37.63%
Recall	61.78%
F1 score	46.77%
ROC-AUC	82.67%
PR-AUC	42.31%


The dataset is imbalanced, with approximately 13.93% of records in the
positive class. Accuracy alone is therefore insufficient to describe
model performance.
Cardiovascular Disease
Final held-out test results:
Metric	Score
Accuracy	86.89%
Precision	85.71%
Recall	85.71%
F1 score	85.71%
ROC-AUC	95.24%
PR-AUC	94.59%


The cardiovascular test set contains only 61 records. These results are
estimates from a small sample and may be unstable; they should not be
directly compared with the diabetes metrics as if the datasets and tasks
were equivalent.
Run Locally
Prerequisites
- Python 3.13.7 or a compatible Python version
- Node.js and npm
- Git (optional, for version control)
1. Clone the repository
git clone https://github.com/OmBhandari05/HealthML.git
cd HealthML
Alternatively, open your existing local project directory.
2. Create a Python environment
On Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, use:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install Python dependencies
python -m pip install -r requirements.txt
Use compatible package versions when loading the saved scikit-learn
models.
4. Start the backend
From the project root:
python -m uvicorn app.main:app --reload
Backend URLs:
- API root: http://127.0.0.1:8000/
- Swagger documentation: http://127.0.0.1:8000/docs
- Available models: http://127.0.0.1:8000/api/v1/models
5. Start the frontend
Open a second terminal:
cd frontend
npm install
npm run dev
Open the frontend URL displayed by Vite, normally
http://localhost:5173/.
Both the backend and frontend must be running for assessments to work.
6. Build and preview the frontend
npm run build
npm run preview
Vite creates a production build in frontend/dist/ and displays a local
preview URL. Keep the backend running and ensure its CORS configuration
permits the preview origin.
API Endpoints
Method	Endpoint	Purpose
GET	/	API status and project information
GET	/api/v1/models	List registered models
POST	/api/v1/predict/diabetes	Generate a diabetes model assessment
POST	/api/v1/predict/cardiovascular	Generate a cardiovascular model assessment


Interactive request schemas and responses are available through Swagger
at /docs.
The prediction response includes the disease identifier, model score,
model assessment, model version, input observations, and educational-use
disclaimer.
Automated Tests
Run the test suite from the project root:
python -m pytest -v
The current test suite contains five tests covering:
- API root response.
- Registered disease models.
- Diabetes prediction endpoint.
- Cardiovascular prediction endpoint.
- Validation of invalid cardiovascular input.
Latest verified result: 5 tests passed. This reflects the test run
performed during development; rerun the suite after making changes.
Limitations and Responsible Use
- This project is for education, experimentation, and demonstration.
- Model scores are not clinically calibrated probabilities.
- Input observations are not necessarily causal explanations of model
  predictions.
- The cardiovascular dataset contains only 303 records.
- Dataset biases and limitations may affect model generalization.
- The models have not been established as safe or effective clinical
  decision-support tools.
- Never use this prototype to diagnose disease, change treatment, or
  delay medical care.
Future Improvements
- Add more disease-specific models through the modular registry.
- Improve model explainability and evaluation reporting.
- Add stronger automated tests for schemas, error handling, and model
  outputs.
- Improve deployment configuration and environment reproducibility.
- Explore external validation and calibration using suitable datasets.
- Add monitoring, logging, and model-version management.
Project Status
Current status: Two trained ML models integrated with a React
frontend and FastAPI backend. Both assessment flows have been tested
locally, the frontend production build succeeds, and the current
automated test suite passes.
The project remains an educational prototype and is not intended for
clinical use.