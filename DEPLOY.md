# Deploying This Project (Free, No Credit Card)

Same approach as the churn project: **Render**, free tier, GitHub-connected, no CLI.

## Steps

1. **Push this project to GitHub**
   ```bash
   cd loan-mlflow-project
   git init
   git add .
   git commit -m "Loan default prediction with MLflow tracking + registry"
   git branch -M main
   git remote add origin https://github.com/Anirudh678/loan-mlflow-project.git
   git push -u origin main
   ```

2. **Create a Render account** at [render.com](https://render.com) (sign in with GitHub).

3. **New Web Service** → connect your `loan-mlflow-project` repo.
   - Render auto-detects `render.yaml` and `Dockerfile`. If not, set manually:
     - **Environment:** Docker
     - **Plan:** Free
     - **Health Check Path:** `/health`

4. **Important difference from the churn project**: this Dockerfile *trains the model during the build* (`RUN python src/generate_data.py && python src/train.py`), so the first build takes longer — expect 3-5 minutes, since it's training 4 separate models and writing them to the MLflow registry inside the image.

5. Once live, test it:
   ```bash
   curl https://loan-default-mlflow-api.onrender.com/health
   curl -X POST https://loan-default-mlflow-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 29, "annual_income": 400000, "loan_amount": 900000, "credit_score": 580, "employment_type": "Self-Employed", "existing_loans": 3, "loan_term_months": 60, "missed_payments_last_year": 2}'
   ```

6. Add the live URL to your resume/README as a **Live Demo** link.

## A Note on This Deployment Approach
Baking training into the Docker build is fine for a portfolio demo where the goal is "prove the pipeline works end-to-end." In a real production setup you would **not** retrain on every deploy — training would be a separate scheduled/triggered job (like the Airflow DAG + CI/CD gate pattern from the MLOps project we scaffolded earlier), and the serving container would just pull the already-registered Production model. Worth mentioning this distinction if it comes up in an interview — it shows you understand *why* the simple version isn't how you'd do it at scale.

## Alternative: Railway
[railway.app](https://railway.app) — same GitHub-connect + Dockerfile auto-detect flow, usage-based free credits instead of Render's sleep/wake behavior.
