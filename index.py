import pandas as pd

# Load the dataset
data = pd.read_csv('CrimePredictionInChicagoDataset.csv')

# Show the initial shape and a quick overview
print("Initial Dataset Shape:", data.shape)
print(data.head())
print(data.info())

# Drop columns that are irrelevant 
irrelevant_columns = ['ID', 'Case Number', 'Block', 'Updated On', 'Location']
# Check if each column exists before attempting to drop it
data = data.drop(columns=[col for col in irrelevant_columns if col in data.columns], axis=1)

# Handle missing values:
# For categorical columns, fill missing values with 'UNKNOWN'
categorical_columns = ['Location Description']
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].fillna('UNKNOWN')

# For numeric columns, fill missing values with the median value of the column
numeric_columns = ['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Ward']
for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# Convert the 'Date' column to datetime format
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Invalid dates become NaT
    # Drop rows where the 'Date' column could not be parsed
    data.dropna(subset=['Date'], inplace=True)

    # Extract useful date-related features from the 'Date' column
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Hour'] = data['Date'].dt.hour
    data['Weekday'] = data['Date'].dt.weekday  # Monday=0, Sunday=6
    data['IsWeekend'] = data['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Drop the original 'Date' column since we've extracted relevant information
    data.drop(columns=['Date'], axis=1, inplace=True)

# Convert binary columns ('Arrest' and 'Domestic') to numeric values (0 or 1)
binary_columns = ['Arrest', 'Domestic']
for col in binary_columns:
    if col in data.columns:
        data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() in ['true', 'yes', '1'] else 0)
