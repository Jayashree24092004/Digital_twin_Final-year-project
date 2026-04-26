🚀 Digital Twin for Astronaut Health Monitoring & Performance Optimization
A machine learning-powered Digital Twin framework that simulates, monitors, and optimizes astronaut physiological health during long-duration space missions.

📌 Overview
Long-duration space missions expose astronauts to microgravity, radiation, and circadian disruption — causing progressive physiological degradation. This project builds a Digital Twin (a virtual replica of astronaut physiology) that continuously predicts health trajectories, detects anomalies early, and recommends personalized countermeasures to maintain peak performance.

✨ Features

🧬 Synthetic Health Data Generation — Simulates realistic astronaut physiological data across 365-day missions with microgravity and radiation effects
📈 Multi-Parameter Health Prediction — Forecasts 7, 30, and 180-day health trajectories using Random Forest Regressor
🚨 Anomaly Detection — Detects physiological anomalies using Isolation Forest on cardiovascular and stress parameters
🏋️ Intervention Simulation — Evaluates three countermeasure plans (Exercise, Balanced, Recovery) and recommends optimal strategies
🤖 Explainable AI — Feature importance analysis to interpret which factors drive health degradation
📊 Rich Visualizations — Health radar charts, degradation plots, intervention heatmaps, and scenario comparisons


🛠️ Tech Stack
ToolPurposePythonCore programming languageNumPyNumerical computationsPandasData handling & preprocessingScikit-learnML models (Random Forest, Isolation Forest)MatplotlibVisualization & plottingSeabornStatistical data visualization

📁 Project Structure
digital-twin-astronaut/
│
├── main.py          # Main source code
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
│
└── outputs/                 # Generated visualizations
    ├── health_degradation.png
    ├── radiation_cognitive.png
    ├── anomaly_detection.png
    ├── feature_importance.png
    ├── scenario_comparison.png
    ├── health_radar.png
    └── intervention_heatmap.png

⚙️ Installation

Clone the repository

bashgit clone(https://github.com/Jayashree24092004/Digital_twin_Final-year-project)
cd digital-twin-astronaut

Install dependencies

bashpip install -r requirements.txt

▶️ Usage
Run the complete Digital Twin pipeline:
bashpython digital_twin.py
This will:

Generate 365 days of synthetic astronaut health data
Train the Digital Twin prediction models
Detect physiological anomalies
Simulate intervention scenarios
Generate and save all visualizations


📊 Monitored Health Parameters
ParameterEffect of MicrogravityMuscle Mass~1–2% loss per monthBone Density~1% loss per monthVO₂ MaxCardiovascular deconditioningHRVAutonomic dysfunctionCognitive ScoreRadiation & stress impactSleep QualityCircadian disruptionStress IndexIncreases with mission durationHeart RateSlight increase over time

🧠 System Architecture
Input Layer          →    Data Processing    →    Digital Twin Model
(Physiological &          (Feature Engineering:    (State Representation +
 Environmental Data)       Lag Features,            Mathematical Modeling)
                           Rolling Mean,
                           Trend Extraction)
        ↓
Prediction Layer     →    Decision Layer     →    Output Layer
(Random Forest)           (Anomaly Detection +     (Health Predictions +
                           Intervention Opt.)       Alerts + Recommendations)

🔬 ML Models Used

Random Forest Regressor — Multi-target health trajectory prediction with lagged time-series features
Isolation Forest — Unsupervised anomaly detection for identifying physiological outlier events


💊 Intervention Plans
PlanFocusOutcomePlan AExerciseImproves muscle mass & cardiovascular fitnessPlan BBalancedMaintains overall physiological stabilityPlan CRecoveryImproves sleep quality & cognitive health

📈 Key Results

Successfully predicts multi-parameter health degradation over 12-month missions
Detects physiological anomalies using unsupervised learning
Demonstrates that Digital Twin-guided interventions outperform the no-intervention baseline
Feature importance analysis reveals mission day and cumulative radiation as the strongest health predictors


🔮 Future Work

Integration with real astronaut biomedical sensor data (NASA HRP datasets)
Real-time health monitoring dashboard
AI-driven automated intervention planning
Integration with space mission simulation platforms
