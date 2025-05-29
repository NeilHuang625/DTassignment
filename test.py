import pandas as pd
import numpy as np

# Read data
filing_df = pd.read_csv("filing_behaviour.csv")
profile_df = pd.read_csv("taxpayer_profiles.csv")

# Data cleaning
# 1. Check and handle missing values
print("Missing values in filing_df:")
print(filing_df.isnull().sum())
print("\nMissing values in profile_df:")
print(profile_df.isnull().sum())

# 2. Handle missing values based on ReturnFiled status
# For records where ReturnFiled is False, PaymentDelayDays and VoluntaryDisclosure should be NA
# For records where ReturnFiled is True, fill missing values appropriately
filing_df['PaymentDelayDays'] = filing_df.apply(
    lambda row: np.nan if not row['ReturnFiled'] else row['PaymentDelayDays'], axis=1
)
filing_df['VoluntaryDisclosure'] = filing_df.apply(
    lambda row: np.nan if not row['ReturnFiled'] else row['VoluntaryDisclosure'], axis=1
)

# Fill remaining missing values for ReturnFiled=True records
filing_df.loc[filing_df['ReturnFiled'], 'PaymentDelayDays'] = filing_df.loc[filing_df['ReturnFiled'], 'PaymentDelayDays'].fillna(0)
filing_df.loc[filing_df['ReturnFiled'], 'VoluntaryDisclosure'] = filing_df.loc[filing_df['ReturnFiled'], 'VoluntaryDisclosure'].fillna(0)

# 3. Handle outliers in numeric columns (only for ReturnFiled=True records)
def remove_outliers(df, column, n_std):
    # Only consider records where ReturnFiled is True
    valid_data = df[df['ReturnFiled']][column].dropna()
    mean = valid_data.mean()
    std = valid_data.std()
    df.loc[df['ReturnFiled'], column] = df.loc[df['ReturnFiled'], column].clip(
        lower=mean - n_std*std, 
        upper=mean + n_std*std
    )
    return df

# Remove outliers from numeric columns
numeric_columns = ['PaymentDelayDays', 'EstimatedTax', 'ActualTax']
for col in numeric_columns:
    filing_df = remove_outliers(filing_df, col, 3)  # 使用3个标准差作为界限

# 4. Ensure data types are correct
filing_df['Year'] = filing_df['Year'].astype(int)
filing_df['ReturnFiled'] = filing_df['ReturnFiled'].astype(int)
filing_df['VoluntaryDisclosure'] = filing_df['VoluntaryDisclosure'].astype(float)  # Changed to float to handle NaN

# Feature engineering
missed = filing_df.groupby('TaxpayerID')[
    'ReturnFiled'].apply(lambda x: 1 - x.mean())
delay = filing_df.groupby('TaxpayerID')['PaymentDelayDays'].mean()
voluntary = filing_df.groupby('TaxpayerID')['VoluntaryDisclosure'].mean()
filing_df['UnderpaymentRatio'] = (
    filing_df['EstimatedTax'] - filing_df['ActualTax']) / filing_df['EstimatedTax']
underpayment = filing_df.groupby('TaxpayerID')['UnderpaymentRatio'].mean()
recent_years = filing_df['Year'].max() - 1
recent_issues = (
    filing_df[filing_df['Year'] >= recent_years]
    .groupby('TaxpayerID')
    .apply(lambda x: int(((~x['ReturnFiled']) | (x['PaymentDelayDays'] > 30)).any()))
)

# Merge features
features_df = pd.DataFrame({
    'MissedFilingsRate': missed,
    'AvgPaymentDelay': delay,
    'VoluntaryDisclosureRate': voluntary,
    'UnderpaymentRatio': underpayment,
    'RecentIssues': recent_issues
}).reset_index()

df = features_df.merge(profile_df, on="TaxpayerID")

# Calculate risk score
def calculate_risk_score(row):
    score = 0
    score += 3 * row['MissedFilingsRate']
    score += 2 * (row['AvgPaymentDelay'] / 180)
    score += 3 * row['UnderpaymentRatio']
    score += 2 * row['RecentIssues']
    score -= 2 * row['VoluntaryDisclosureRate']
    return score


df['RiskScore'] = df.apply(calculate_risk_score, axis=1)
df['RiskLevel'] = pd.qcut(df['RiskScore'], q=3, labels=[
                          "Low", "Medium", "High"])

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display all results
print(df[['TaxpayerID', 'RiskScore', 'RiskLevel']])
