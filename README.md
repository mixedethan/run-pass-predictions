# NFL Run/Pass Predictor

A machine learning application that predicts whether an NFL offense will call a run or pass play based on real-time game situations. 

## Project Overview
This tool uses historical NFL play-by-play data (sourced from `nflfastR`) to train a LightGBM classification model. It analyzes factors like down, distance, score differential, and field position to output play-call probabilities. It then takes these probabilities displays them in a Streamlit web application for ease of understanding

**Key Features:**
- **Real-time Prediction:** Adjust sliders to simulate any game scenario.
- **Decision Explanations:** Uses SHAP values to visualize *why* the model predicts a specific play.
- **4th Down Logic:** Separate model for 4th down decision making (Go for it vs. Kick/Punt).
- **Interactive Field:** Custom Matplotlib visualization of the field with an interactive line of scrimmage and first down line.

## Tech Stack
- **Frontend:** Streamlit
- **Modeling:** LightGBM, Scikit-learn
- **Explainability:** SHAP
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib

## Installation

1. Clone the repo:
   ```bash
   git clone [https://github.com/mixedethan/run-pass-predictions.git](https://github.com/mixedethan/run-pass-predictions.git)
   cd run-pass-predictions