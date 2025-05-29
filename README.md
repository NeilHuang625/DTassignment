# Taxpayer Risk Segmentation Prototype

## Goal

To develop a simple, interpretable prototype that:

- Calculates taxpayer risk scores based on behavioral indicators
- Segments taxpayers into risk categories

## Contents

- `test.py` – Main script for data processing(loading, cleaning, feature engineering, and merging datasets), feature engineering, and scoring logic
- `filing_behaviour.csv` – Simulated taxpayer filing records (3 years)
- `taxpayer_profiles.csv` – Basic taxpayer demographic information

## Risk Scoring Logic

Five indicators were derived from the data:

- **Missed Filings Rate**
- **Average Payment Delay**
- **Underpayment Ratio**
- **Recent Issues (last 2 years)**
- **Voluntary Disclosure Rate**

Each indicator was assigned a weight based on domain intuition and expected risk impact. A composite risk score was calculated per taxpayer, and they were segmented into `Low`, `Medium`, and `High` risk groups using quantile-based thresholds.

## Future Improvements

- Incorporate real audit outcomes to train supervised models
- Build a working front-end in React
- Add filtering by location or taxpayer type
