"""
End-to-end training pipeline with full MLflow experiment tracking:
- Logs params/metrics/artifacts for every run
- Compares multiple models + hyperparameter configs as separate MLflow runs
- Registers the best model to the MLflow Model Registry with a gate (min AUC)

Run: python src/train.py
Then view results: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

MODEL_NAME = "loan-default-classifier"
MIN_AUC_TO_REGISTER = 0.75

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan-default-prediction")

df = pd.read_csv("data/loan_data.csv")
X = df.drop(columns=["defaulted"])
y = df["defaulted"]

numeric_features = ["age", "annual_income", "loan_amount", "credit_score",
                     "existing_loans", "loan_term_months", "missed_payments_last_year"]
categorical_features = ["employment_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Each entry is one MLflow run: a model type + a hyperparameter configuration
experiment_configs = [
    {"name": "logreg_default", "model": LogisticRegression(max_iter=1000), "params": {"max_iter": 1000}},
    {"name": "rf_shallow", "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
     "params": {"n_estimators": 100, "max_depth": 5}},
    {"name": "rf_deep", "model": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
     "params": {"n_estimators": 300, "max_depth": 12}},
    {"name": "gbm_default", "model": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
     "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}},
]

best_run_auc = 0
best_run_id = None
best_config_name = None

for config in experiment_configs:
    with mlflow.start_run(run_name=config["name"]) as run:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", config["model"])])
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = pipeline.predict(X_test)

        metrics = {
            "auc": roc_auc_score(y_test, probs),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

        mlflow.log_params(config["params"])
        mlflow.log_param("model_type", type(config["model"]).__name__)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"[{config['name']}] AUC={metrics['auc']:.4f} "
              f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f}")

        if metrics["auc"] > best_run_auc:
            best_run_auc = metrics["auc"]
            best_run_id = run.info.run_id
            best_config_name = config["name"]

print(f"\nBest run: {best_config_name} (AUC={best_run_auc:.4f}, run_id={best_run_id})")

# Gate: only register to the Model Registry if it clears the bar
if best_run_auc >= MIN_AUC_TO_REGISTER:
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered '{MODEL_NAME}' version {result.version} to the Model Registry.")

    # Promote it to the Production stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Version {result.version} promoted to Production stage.")
else:
    print(f"Best AUC {best_run_auc:.4f} did not clear the {MIN_AUC_TO_REGISTER} threshold — not registered.")
