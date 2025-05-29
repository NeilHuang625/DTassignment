import pandas as pd

# Read data
filing_df = pd.read_csv("filing_behaviour.csv")
profile_df = pd.read_csv("taxpayer_profiles.csv")

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
