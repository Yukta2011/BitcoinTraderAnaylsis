import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, f_oneway

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

sns.set_style("whitegrid")


# LOAD DATA

fear = pd.read_csv("fear.csv")
trades = pd.read_csv("historical_data.csv")


# CLEAN SENTIMENT DATA

fear['date'] = pd.to_datetime(fear['date'])

fear.rename(columns={
    'classification': 'Classification'
}, inplace=True)


# CLEAN TRADER DATA

trades.rename(columns={
    'Account': 'account',
    'Side': 'side',
    'Closed PnL': 'closedPnL',
    'Size USD': 'size'
}, inplace=True)

# Convert timestamp

trades['Timestamp'] = pd.to_datetime(
    trades['Timestamp'],
    unit='ms'
)

trades['date'] = trades['Timestamp'].dt.date
trades['date'] = pd.to_datetime(trades['date'])

# MERGE


merged = pd.merge(
    trades,
    fear[['date', 'Classification', 'value']],
    on='date',
    how='left'
)

print("Merged Shape:", merged.shape)

# CREATE TARGET


merged['win'] = np.where(
    merged['closedPnL'] > 0,
    1,
    0
)


# PROFIT ANALYSIS


profit_sentiment = merged.groupby(
    'Classification'
)['closedPnL'].mean()

print("\nAverage Profit")
print(profit_sentiment)


# WIN RATE


win_rates = merged.groupby(
    'Classification'
)['win'].mean() * 100

print("\nWin Rate")
print(win_rates)


# BUY VS SELL


buy_sell = merged.groupby(
    ['Classification', 'side']
)['closedPnL'].mean()

print("\nBuy vs Sell")
print(buy_sell)

# POSITION SIZE


size_analysis = merged.groupby(
    'Classification'
)['size'].mean()

print("\nAverage Position Size")
print(size_analysis)

# CORRELATION


numerical = [
    'closedPnL',
    'size',
    'value'
]

corr = merged[numerical].corr()

print("\nCorrelation Matrix")
print(corr)


# T TEST


fear_trades = merged[
    merged['Classification'] == 'Fear'
]['closedPnL']

greed_trades = merged[
    merged['Classification'] == 'Greed'
]['closedPnL']

t_stat, p_value = ttest_ind(
    fear_trades,
    greed_trades,
    equal_var=False
)

print("\nT Test")
print("T-stat:", t_stat)
print("P-value:", p_value)


# ANOVA


groups = []

for sentiment in merged[
    'Classification'
].dropna().unique():

    groups.append(
        merged[
            merged['Classification']
            == sentiment
        ]['closedPnL']
    )

f_stat, p_val = f_oneway(*groups)

print("\nANOVA")
print("F-stat:", f_stat)
print("P-value:", p_val)


# TOP TRADERS


top_traders = merged.groupby(
    'account'
).agg({
    'closedPnL': 'sum',
    'size': 'mean'
})

top_traders = top_traders.sort_values(
    by='closedPnL',
    ascending=False
)

print("\nTop Traders")
print(top_traders.head(10))


# MACHINE LEARNING


le_side = LabelEncoder()
le_sentiment = LabelEncoder()

merged['side_enc'] = le_side.fit_transform(
    merged['side']
)

merged['sentiment_enc'] = le_sentiment.fit_transform(
    merged['Classification']
)

features = [
    'size',
    'value',
    'side_enc',
    'sentiment_enc'
]

X = merged[features]

y = merged['win']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy")
print(
    accuracy_score(
        y_test,
        pred
    )
)

print(
    classification_report(
        y_test,
        pred
    )
)

# FEATURE IMPORTANCE


importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)




plt.figure(figsize=(8,5))
sns.barplot(
    x=profit_sentiment.index,
    y=profit_sentiment.values
)
plt.title("Average PnL by Sentiment")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(
    x=win_rates.index,
    y=win_rates.values
)
plt.title("Win Rate by Sentiment")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(
    data=merged,
    x='Classification',
    y='closedPnL',
    hue='side'
)
plt.title("Buy vs Sell Performance")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()