"""
Medical Cost Predictor - Gradio Web App
AI/ML Build Week - Day 7 Assignment
"""

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load model artifact
artifact = joblib.load("best_model.pkl")
model    = artifact["model"]
features = artifact["features"]
median_charges = artifact["median_charges"]

# Keep a simple in-memory prediction history
history: list[dict] = []


def predict(age, sex, bmi, children, smoker, region):
    """Run inference and return a formatted result + updated history table."""

    # Build input dict matching training columns
    input_dict = {
        "age":      age,
        "sex":      1 if sex == "Male" else 0,
        "bmi":      bmi,
        "children": children,
        "smoker":   1 if smoker == "Yes" else 0,
    }

    # One-hot region (same scheme as pd.get_dummies with drop_first=True)
    region_cols = {
        "region_northwest": 0,
        "region_southeast": 0,
        "region_southwest": 0,
    }
    key = f"region_{region.lower()}"
    if key in region_cols:
        region_cols[key] = 1
    # "northeast" is the dropped baseline → all zeros

    input_dict.update(region_cols)

    # Align to training feature order
    input_df = pd.DataFrame([input_dict])[features]

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    label    = "HIGH COST" if pred == 1 else "LOW COST"
    prob_pct = f"{proba:.1%}"

    result = (
        f"## Prediction: {label}\n\n"
        f"**Probability of High Cost:** {prob_pct}\n\n"
        f"*Threshold based on median insurance charge: ${median_charges:,.0f}*"
    )

    # Append to history
    history.append({
        "Time":      datetime.now().strftime("%H:%M:%S"),
        "Age":       age,
        "BMI":       round(bmi, 1),
        "Smoker":    smoker,
        "Region":    region,
        "Prediction":label,
        "Probability": prob_pct,
    })

    history_df = pd.DataFrame(history[::-1])  # newest first

    return result, history_df


# UI Layout 
with gr.Blocks(title="Medical Cost Predictor", theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # Medical Cost Predictor
        ### Predicts whether your insurance charges will be **High Cost** or **Low Cost**
        Fill in your details below and click **Predict**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            age      = gr.Slider(18, 65, value=30, step=1, label="Age")
            sex      = gr.Radio(["Male", "Female"], label="Sex", value="Male")
            bmi      = gr.Slider(10.0, 55.0, value=25.0, step=0.1, label="BMI")
            children = gr.Slider(0, 5, value=0, step=1, label="Number of Children")
            smoker   = gr.Radio(["Yes", "No"], label="Smoker?", value="No")
            region   = gr.Dropdown(
                choices=["Northeast", "Northwest", "Southeast", "Southwest"],
                value="Northeast",
                label="Region"
            )
            btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            output = gr.Markdown(label="Result")

    gr.Markdown("---\n### Prediction History")
    history_table = gr.Dataframe(
        headers=["Time","Age","BMI","Smoker","Region","Prediction","Probability"],
        label="",
        interactive=False,
    )

    btn.click(
        fn=predict,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=[output, history_table],
    )

    gr.Markdown(
        "_Model trained on the UCI Medical Insurance dataset. "
        "For educational purposes only._"
    )

if __name__ == "__main__":
    app.launch()
