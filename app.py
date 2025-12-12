import gradio as gr
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_salary(years_experience: float) -> float:
    df = pd.DataFrame({'YearsExperience': [years_experience]})
    pred = model.predict(df)[0]
    return round(float(pred), 2)

demo = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(label="Years of Experience"),
    outputs=gr.Number(label="Predicted Salary"),
    title="Salary Prediction App",
    description="Enter years of experience to predict salary using a trained Linear Regression model."
)

if __name__ == "__main__":
    demo.launch()
