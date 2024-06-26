import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/nayan/OneDrive/Desktop/Data Visualization/retraction.csv')

# List of columns to drop
data.drop(columns=[
    'Notes', 'URLS', 'OriginalPaperPubMedID', 'OriginalPaperDOI',
    'RetractionDOI', 'RetractionPubMedID', 'Record ID', 'RetractionNature', 'Title'
], inplace=True)

# Convert date columns to datetime specifying the correct format
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%d/%m/%Y')
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%d/%m/%Y')

# Calculate the number of days between the dates
data['DaysBetween'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days

# Convert all text data to lowercase and replace various separators with commas
for column in data.select_dtypes(include=[object]):
    data[column] = data[column].str.lower()  # Convert text to lowercase
    data[column] = data[column].str.replace(r'[;:]+', ',', regex=True)  # Replace semicolons and colons with commas

# Remove rows with any missing values
data = data.dropna()

# Save the cleaned data to a new CSV file
data.to_csv('C:/Users/nayan/OneDrive/Desktop/Data Visualization/clean.csv', index=False)
