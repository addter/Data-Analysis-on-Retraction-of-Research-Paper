
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from your CSV file
df = pd.read_csv("C:/Users/nayan/OneDrive/Desktop/Data Visualization/clean.csv")
# Convert infinite values to NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)



# Convert date columns to datetime
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'])
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'])

# Function to get a list of reasons without exploding
def get_reasons_list(reasons_series):
    reasons_list = []
    for reason_string in reasons_series:
        if reason_string:
            reasons_list.extend(reason.strip() for reason in reason_string.split('+') if reason.strip())
    return reasons_list

# Get a list of all reasons from the 'Reason' column
all_reasons = get_reasons_list(df['Reason'])

# Count the frequency of each reason
reason_freq = pd.Series(all_reasons).value_counts()

# Frequency of Retractions over Time
plt.figure(figsize=(12, 6))
retraction_counts = df.groupby(df['RetractionDate'].dt.to_period('M')).size()
retraction_counts.plot(kind='line')
plt.title('Frequency of Retractions over Time')
plt.xlabel('Date')
plt.ylabel('Number of Retractions')
plt.grid(True)
plt.show()

# Visualization - now with a manageable number of reasons
top_n = 10
top_reasons = reason_freq.head(top_n)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_reasons.index, x=top_reasons.values)
plt.title('Top Reasons for Retractions')
plt.xlabel('Frequency')
plt.ylabel('Reason')
plt.tight_layout()
plt.show()

# Citation Count Trends
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Paywalled', y='CitationCount')
plt.title('Citation Count Trends by Paywalled Status')
plt.xlabel('Paywalled Status')
plt.ylabel('Citation Count')
plt.show()

# Time Between Publication and Retraction
df['TimeToRetraction'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='TimeToRetraction', bins=20, kde=True)
plt.title('Time Between Publication and Retraction')
plt.xlabel('Time (Days)')
plt.ylabel('Frequency')
plt.show()

# Geographical Trends
plt.figure(figsize=(10, 6))
country_counts = df['Country'].value_counts().head(10)
country_counts.plot(kind='bar')
plt.title('Top 10 Countries with Retractions')
plt.xlabel('Country')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.show()

# Journal and Publisher Analysis
plt.figure(figsize=(10, 6))
journal_counts = df['Journal'].value_counts().head(10)
journal_counts.plot(kind='bar')
plt.title('Top 10 Journals with Retractions')
plt.xlabel('Journal')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.show()

# Author Analysis
plt.figure(figsize=(10, 6))
author_counts = df['Author'].value_counts().head(10)
author_counts.plot(kind='bar')
plt.title('Top 10 Authors with Retractions')
plt.xlabel('Author')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.show()

# Article Type Analysis
plt.figure(figsize=(10, 6))
article_type_counts = df['ArticleType'].value_counts().head(10)
article_type_counts.plot(kind='bar')
plt.title('Top 10 Article Types with Retractions')
plt.xlabel('Article Type')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.show()
