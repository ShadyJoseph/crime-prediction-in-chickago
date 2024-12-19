import random

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import ttk, messagebox

original_data = pd.read_csv('Crime Prediction in Chicago_Dataset.csv')

# Specify columns that you want to keep for training
relevant_columns = ['Year', 'Month', 'Day', 'Day of Week', 'Latitude', 'Longitude', 'Domestic', 'Primary Type',
                    'Location Description', 'Description', 'Beat', 'Arrest']

input_features = ['Year', 'Month', 'Day', 'Day of Week', 'Latitude', 'Longitude', 'Domestic', 'Primary Type',
                  'Location Description', 'Description', 'Beat']

# Dictionary to store LabelEncoders for categorical columns
label_encoders = {}


# Function to remove outliers using IQR
def remove_outliers(df, columns, multiplier=1.5):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# Function to preprocess data
def preprocess_data():
    data = pd.read_csv('Crime Prediction in Chicago_Dataset.csv')

    # Check if 'Date' column exists
    if 'Date' in data.columns:
        # Process the 'Date' column to create new features: Year, Month, Day, Day of Week
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid date
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Day of Week'] = data['Date'].dt.weekday
        data.drop(columns=['Date'], axis=1, inplace=True)  # Drop the 'Date' column after extracting needed info
    else:
        # If 'Date' is not available, handle the missing columns gracefully
        print("Warning: 'Date' column not found. Skipping date-related columns.")

    # Keep only the relevant columns for training
    data = data[relevant_columns]

    # Handle missing categorical columns
    categorical_columns = data.select_dtypes(include='object').columns
    for col in categorical_columns:
        data[col] = data[col].fillna('UNKNOWN')

    # Handle missing numeric columns by replacing NaN with median
    numeric_columns = data.select_dtypes(include='number').columns
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())

    # Remove outliers
    data = remove_outliers(data, numeric_columns)

    # Convert 'Domestic' and 'Arrest' to binary (0 or 1)
    binary_columns = ['Arrest', 'Domestic']
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() in ['true', 'yes', '1'] else 0)

    # Label encoding for categorical variables and storing encoders
    for col in categorical_columns:
        if col != 'Beat':
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])
            label_encoders[col] = label_encoder  # Store the encoder for each column

    return data, categorical_columns


# Function to evaluate model performance
def evaluate_models(X_test, y_test, models):
    """
    This function evaluates models and prints the evaluation metrics.
    Arguments:
    - X_test: Features of the test dataset.
    - y_test: True labels of the test dataset.
    - models: Dictionary of models to evaluate, with model names as keys and model objects as values.
    """
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        # Print the evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{confusion}\n")


# Train models
def train_models():
    global logistic_model, decision_tree_model, knn_model, svm_model, data, svm_model_cal

    # Preprocess the dataset
    data, categorical_columns = preprocess_data()

    # Prepare the features (X) and target (y)
    X = data.drop(columns=['Arrest'])  # Features (drop target 'Arrest')
    y = data['Arrest']  # Target (Arrest)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Logistic Regression
    logistic_model = LogisticRegression(solver="liblinear", class_weight="balanced", C=10, random_state=0)
    logistic_model.fit(X_train, y_train)

    # Train Decision Tree
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)

    # Train K-Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    # Train Linear SVM
    svm_model = LinearSVC(random_state=42, max_iter=10000, class_weight="balanced", C=10.0)
    svm_model.fit(X_train, y_train)  # Fit the SVM first

    # # Now calibrate the SVM using CalibratedClassifierCV
    # svm_model_cal = CalibratedClassifierCV(svm_model, method="sigmoid", cv="prefit")  # Calibrate after fitting
    # svm_model_cal.fit(X_train, y_train)  # Calibrate (fit) the calibrated model

    print("Models trained successfully!")
    # Now evaluate the models
    models = {
        'Logistic Regression': logistic_model,
        'Decision Tree': decision_tree_model,
        'K-Nearest Neighbors': knn_model,
        'Linear SVM': svm_model
    }

    # Assuming you have X_test and y_test as your test data
    evaluate_models(X_test, y_test, models)


# Function to predict arrest based on user input
def predict_arrest():
    # Get the features entered by the user
    input_data = []
    for feature in input_features:
        feature_value = input_fields[feature].get()

        # For numerical columns (Latitude, Longitude, etc.), convert directly to float
        try:
            # Convert to float if possible, otherwise keep as string for categorical values
            if feature in ['Latitude', 'Longitude', 'Year', 'Month', 'Day', 'Day of Week']:
                input_data.append(float(feature_value))  # Numeric values as float
            else:
                input_data.append(feature_value)  # For categorical fields
        except ValueError:
            input_data.append(feature_value)

    # Handling the 'Domestic' field separately to ensure binary conversion
    if input_data[input_features.index('Domestic')] not in ['Yes', 'No']:
        messagebox.showerror("Invalid Input", "Please enter 'Yes' or 'No' for the 'Domestic' field.")
        return

    # Convert 'Domestic' to binary (1 for Yes, 0 for No)
    input_data[input_features.index('Domestic')] = 1 if input_data[input_features.index('Domestic')] == 'Yes' else 0

    # Convert to DataFrame for prediction
    input_data = pd.DataFrame([input_data], columns=input_features)

    # Preprocess categorical data (encode categorical inputs)
    for col in input_data.select_dtypes(include=['object']).columns:
        try:
            # If the column is categorical, use the label encoder to transform it
            if col != 'Beat':
                input_data[col] = label_encoders[col].transform(input_data[col])
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"The value for {col} is invalid. Please select a valid option.")
            return

    # Predict with the selected model
    model_name = model_var.get()

    if model_name == 'Logistic Regression':
        model = logistic_model
        arrest_prob = model.predict_proba(input_data)[:, 1][0]  # Get probability of arrest
        arrest_pred = model.predict(input_data)[0]  # Get the predicted arrest
    elif model_name == 'Decision Tree':
        model = decision_tree_model
        # Get the probability of the positive class (arrest=1)
        arrest_prob = model.predict_proba(input_data)[:, 1][0]  # Get probability of arrest
        # Get the predicted arrest (0 or 1)
        arrest_pred = model.predict(input_data)[0]  # Get the predicted arrest
    elif model_name == 'K-Nearest Neighbors':
        model = knn_model
        arrest_prob = model.predict_proba(input_data)[:, 1][0]  # Get probability of arrest
        arrest_pred = model.predict(input_data)[0]  # Get the predicted arrest
    elif model_name == 'Linear SVM':
        model = svm_model
        arrest_pred = model.predict(input_data)[0]  # Get the predicted arrest
        if(arrest_pred == 1):
            arrest_prob = 1
        else:
            arrest_prob = 0

    # Show the result
    result_text = f"Predicted Arrest: {arrest_pred}\nProbability of Arrest: {arrest_prob:.2f}"
    result_label.config(text=result_text)


def fill_sample_data(input_fields):
    # Example sample data dictionary with fixed values for some fields
    sample_data = {
        'Year': str(2022),
        'Month': str(random.randint(1, 12)),  # Random month
        'Day': str(random.randint(1, 31)),  # Random day
        'Day of Week': str(random.randint(0, 6)),  # Random day of the week (0=Sunday, 6=Saturday)
        'Latitude': str(random.uniform(41.0, 42.0)),  # Random latitude
        'Longitude': str(random.uniform(-88.0, -87.0)),  # Random longitude
        'Domestic': random.choice(['Yes', 'No']),  # Random 'Yes' or 'No'
        'Primary Type': random.choice(original_data['Primary Type'].unique().tolist()),  # Random crime type
        'Location Description': random.choice(original_data['Location Description'].unique().tolist()),
        'Description': random.choice(original_data['Description'].unique().tolist()),  # Random description
        'Beat': str(random.randint(1, 3000))  # Random Beat code
    }

    # Populate the input fields with the sample data
    for feature, value in sample_data.items():
        if feature in input_fields:  # Check if the feature exists in input_fields
            input_field = input_fields[feature]

            # If it's a Combobox, use set()
            if isinstance(input_field, ttk.Combobox):
                input_field.set(value)
            # If it's an Entry, use insert()
            elif isinstance(input_field, tk.Entry):
                input_field.delete(0, tk.END)  # Clear the current value
                input_field.insert(0, value)  # Insert the sample value
            # You can extend this to handle other types of widgets if necessary (e.g., Checkboxes)
            else:
                print(f"Unknown input field type for {feature}")
        else:
            print(f"Field {feature} not found in input_fields")


# Initialize the main tkinter window
root = tk.Tk()
root.title('Crime Prediction Model')

# Train all models when the app starts
train_models()

# UI elements for model selection
model_var = tk.StringVar(value='Logistic Regression')
model_label = tk.Label(root, text="Select Model:")
model_label.pack(pady=5)

model_dropdown = ttk.Combobox(root, textvariable=model_var,
                              values=['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Linear SVM'])
model_dropdown.pack(pady=5)

# Input fields for user features
input_frame = tk.Frame(root)
input_frame.pack(pady=20)

input_fields = {}  # Dictionary to store input fields

for feature in input_features:
    label = tk.Label(input_frame, text=feature)
    label.grid(row=input_features.index(feature), column=0, padx=5, pady=5)

    if feature in ['Primary Type', 'Location Description', 'Description', 'Beat']:  # Categorical features
        input_fields[feature] = ttk.Combobox(input_frame, values=original_data[feature].unique().tolist())
    elif feature == 'Domestic':
        input_fields[feature] = ttk.Combobox(input_frame, values=["Yes", "No"])
    else:  # Numeric features
        input_fields[feature] = tk.Entry(input_frame)

    input_fields[feature].grid(row=input_features.index(feature), column=1, padx=5, pady=5)

# Button to predict arrest
predict_button = tk.Button(root, text="Predict Arrest", command=predict_arrest)
predict_button.pack(pady=20)

# Result label to show the prediction
result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 12))
result_label.pack(pady=20)

# Button to fill the sample data
sample_button = tk.Button(root, text="Fill Sample Data", command=lambda: fill_sample_data(input_fields))
sample_button.pack(pady=10)

# Run the application
root.mainloop()
