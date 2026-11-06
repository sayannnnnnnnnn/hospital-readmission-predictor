# 30-Day Hospital Readmission Risk Predictor

An interactive web app that ranks diabetic patients by their risk of being readmitted
within 30 days of discharge, so a care team can focus follow-up on the highest-risk group.

**Live app:** [diabetic-readmission-risk-sayan.streamlit.app](https://diabetic-readmission-risk-sayan.streamlit.app/) 

## Results

Data: UCI "Diabetes 130-US hospitals" dataset, reduced to 69,990 patients
(one encounter per patient, deaths and hospice discharges removed).
Only about 9% of patients are readmitted within 30 days, so accuracy is not a useful metric.

| Model | Test ROC-AUC | Test PR-AUC | Top-20% capture |
|---|---|---|---|
| Random Forest | 0.653 | 0.178 | 38.4% |
| Logistic Regression | 0.649 | 0.170 | 37.6% |
| Decision Tree | 0.624 | 0.154 | 36.5% |
| KNN | 0.570 | 0.111 | 27.0% |

Following up with the 20% of patients the model ranks riskiest reaches about 38% of all
30-day readmissions, roughly 1.9x better than picking patients at random.

Logistic Regression is the recommended model: it scores about the same as Random Forest on
new patients, overfits far less (train-test ROC-AUC gap of 0.01 vs 0.20), and its weights
can be read directly.

## What drives readmission

- **Prior inpatient visits are the clearest warning sign.** Patients with no earlier inpatient stay were readmitted about 8% of the time, versus about 13% with one prior stay, 19% with two and 24% with three. It is also the largest numeric weight in the logistic regression model.
- **Where the patient goes after discharge matters a lot.** Patients sent home (44,322 of them) were readmitted 6.9% of the time. Patients sent to rehab or another inpatient care institution were readmitted at roughly 21-26%, about 3-4x as often. Groups with only a handful of patients were ignored as noise.
- **Age and primary diagnosis are weak signals.** Risk rises from about 2% in the youngest patients to about 10-11% at ages 70-90, and diagnosis groups only range from about 7% to 11%. These are associations, not proven causes: a rehab discharge likely marks a sicker patient rather than causing readmission.

## How it works

1. Cleaned the data: removed patients who died or went to hospice, kept one visit per patient
   to avoid train/test leakage, dropped mostly-empty columns, grouped diagnosis codes.
2. Preprocessing pipeline (impute, scale, one-hot encode) fit on training data only.
3. Trained and compared four models with class weighting for the 9% imbalance.
4. Evaluated with ROC-AUC, PR-AUC and top-20% capture instead of accuracy.
5. The app shows each patient's risk as a percentile rank with Low / Medium / High tiers,
   because raw probabilities are not comparable across models.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `readmission_analysis.ipynb`: full analysis notebook
- `app.py`: Streamlit app
- `models/`: saved model pipelines and app data files

## Limitations

A learning project, not a clinical tool. Readmission is hard to predict (ROC-AUC around 0.65),
and the data covers 130 US hospitals from 1999 to 2008.

Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008