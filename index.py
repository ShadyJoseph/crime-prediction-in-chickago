import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# Function to remove outliers using IQR
def remove_outliers(df, columns, multiplier=1.5):
    """
    Removes outliers from specified numeric columns in the DataFrame using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of numeric column names to check for outliers.
        multiplier (float): The multiplier for the IQR range (default is 1.5).
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            # Keep rows within the bounds
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Load the dataset
data = pd.read_csv('CrimePredictioninChicagoDataset.csv')

print("\n_______________________________________________________________\n")
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


print("\n_______________________________________________________________\n")
# Remove duplicates
print("Duplicates before:", data.duplicated().sum())
data = data.drop_duplicates()
print("Duplicates after:", data.duplicated().sum())

# For numeric columns, fill missing values with the median value of the column
numeric_columns = ['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Ward']
for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# Remove outliers from numeric columns
data = remove_outliers(data, numeric_columns)


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

# Label Encoding for Categorical Columns
# Identify remaining non-numeric columns for label encoding
label_encoder = LabelEncoder()
categorical_columns_to_encode = data.select_dtypes(include=['object']).columns

for col in categorical_columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

print("\n_______________________________________________________________\n")
# Show the final dataset structure
print("Final Dataset Shape:", data.shape)
print(data.head())
print(data.info())

#end of pre-processing



#Logistic regression
# Selecting x (features) and y (target)
x = data.loc[:, data.columns != 'Arrest']
y = data['Arrest']


model = LogisticRegression(solver="liblinear",class_weight="balanced", C=10, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
p_pred = model.predict_proba(x_test)
score_ = model.score(x_test, y_test)
conf_m = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n_______________________________________________________________\n")
print("Logistic regression")
print(conf_m)
print(score_)
print(report)


print("\n_______________________________________________________________\n")
##### CLASSIFICATION ALGORITHM DECISION TREE ####

print("\n\t\t\tDECISION TREE MODEL: ")

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(x_test)

# Metrics for evaluating the model
print("Metrics for evaluating this model:")
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)


print("\n_______________________________________________________________\n")
##### CLASSIFICATION ALGORITHM KNN ####

print("\n\t\t\tK-NEAREST NEIGHBORS MODEL (KNN): ")
# Create a K-Nearest Neighbors classifier
clf2 = KNeighborsClassifier()

# Train the classifier on the training data
clf2.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf2.predict(x_test)

# Metrics for evaluating the model
print("Metrics for evaluating the K-Nearest Neighbors (KNN) model:")
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)


print("\n_______________________________________________________________\n")

# Initialize the Linear SVM model
linear_svm_model = LinearSVC(random_state=42, max_iter=10000, class_weight="balanced")

# Train the model
linear_svm_model.fit(x_train, y_train)

# Make predictions
y_pred = linear_svm_model.predict(x_test)

# Evaluate the model
print("Linear SVM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))