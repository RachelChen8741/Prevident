from flask import Flask, request, render_template
import pandas as pd
import joblib
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


app = Flask(__name__)


classifier = joblib.load("random_forest_classifier_compressed.pkl")
regressor = joblib.load("random_forest_regressor_compressed.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Use the appropriate template name


@app.route('/predictform')
def predictform():
    return render_template('predictform.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Standard inputs
        input_data = {
            'Offered Food': [1 if request.form['Offered_Food'] == 'Yes' else 0],
            'Expected': [int(request.form['Expected'])],  # Ensure this input exists and is an integer
            'Org Size': [request.form['Org_Size']],
            'Type': [request.form['Type']]
        }

        # Multi-select options setup
        multi_select_options = {
            'Days': 'Days[]',
            'Time': 'Time[]',
            'Advertisements': 'Advertisements[]'
        }

        print("huh", file=sys.stderr)
        # Process multi-select inputs; create binary columns for each option
        for field_prefix, form_name in multi_select_options.items():
            print("are you getting to here", file=sys.stderr)
            options = request.form.getlist(field_prefix)
            print(options, file=sys.stderr)
            for option in options:
                print("what " + option, file=sys.stderr)
                input_data[f'{field_prefix}_{option}'] = [True]  # Example: 'Days_Monday': [1]
                
        # print("where", file=sys.stderr)
        input_df = pd.DataFrame(input_data)
        print(input_df, file=sys.stderr)

        org_size = [
        'Org Size_Small (1-50 members)',
        'Org Size_Medium (50-200 members)',
        'Org Size_Large (200+ members)'
    ]

    # Create new features and preprocess data
    # input_df["Popular"] = input_df["Actual"] >= input_df["Expected"]
    # input_df["Popular"] = input_df["Popular"].astype(int)  # Convert boolean to integer
        # for feature in org_size:
        #     if feature not in input_df.columns:
        #         # Add missing features as columns of zeros
        #         input_df[feature] = False
        #         print(input_df, file=sys.stderr)
        # input_df = confusion(input_df)
        # print(input_df, file=sys.stderr)

        # Ensure the order of columns matches the training data
        one_hot_cols = ['Org Size', 'Type']
        input_df = pd.get_dummies(input_df, columns=one_hot_cols)
        
        for feature in regressor.feature_names_in_:
            # print("here?", file=sys.stderr)
            if feature not in input_df.columns:
                # Add missing features as columns of zeros
                input_df[feature] = False
                # print(input_df, file=sys.stderr)
                # print("is it here", file=sys.stderr)
        input_df = input_df[regressor.feature_names_in_]
        # print("tired", file=sys.stderr)
        print(input_df, file=sys.stderr)
        input_df.to_csv('full_dataframe.csv', index=False)

        # Prediction
        prediction = regressor.predict(input_df)[0]
        return render_template('prediction.html', prediction_text=f'Predicted number of attendees: {prediction:.0f}')
    except Exception as e:
        print(e, file=sys.stderr)  # Output the error to stderr
        return render_template('prediction.html', prediction_text=f'Error: {str(e)}')
    
def confusion(input_df):

    # Perform one-hot encoding on 'Size of Organization' and 'Type of Event'
    one_hot_cols = ['Org Size', 'Type']
    input_df = pd.get_dummies(input_df, columns=one_hot_cols)
    

    org_size = [
        'Org Size_Small (1-50 members)',
        'Org Size_Medium (50-200 members)',
        'Org Size_Large (200+ members)'
    ]

    # Create new features and preprocess data
    # input_df["Popular"] = input_df["Actual"] >= input_df["Expected"]
    # input_df["Popular"] = input_df["Popular"].astype(int)  # Convert boolean to integer
    for feature in org_size:
            if feature not in input_df.columns:
                # Add missing features as columns of zeros
                input_df[feature] = 0


    # Remove unnecessary columns and prepare data for modeling
    input_df = input_df.drop(['Expected'], axis=1)
    print('mom', file=sys.stderr)
    
    return input_df


@app.route('/classifyform')
def classifyform():
    return render_template('classifyform.html')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Preprocess the input data
        input_data = {
            'Offered Food': [1 if request.form['Offered_Food'] == 'Yes' else 0],
            'Org Size': [request.form['Org_Size']],
            'Type': [request.form['Type']],
            'Expected': [int(request.form['Expected'])]
        }

        multi_select_options = {
            'Days': 'Days[]',
            'Time': 'Time[]',
            'Advertisements': 'Advertisements[]'
        }

        # Adding entries for multi-select fields initialized to 0
        for field_prefix, form_name in multi_select_options.items():
            options = request.form.getlist(field_prefix)
            for option in options:
                input_data[f'{field_prefix} - {option}'] = [
                    1]  # Example: 'Days_Monday': [1]
        
        input_df = pd.DataFrame(input_data)
        input_df = preprocess_data(input_df)  # Call your preprocessing function
        for feature in classifier.feature_names_in_:
            if feature not in input_df.columns:
                # Add missing features as columns of zeros
                input_df[feature] = 0
        input_df = input_df[classifier.feature_names_in_]
        input_df = input_df.fillna(0)
        input_df = input_df * 1
        
        # Predict using the model
        prediction = classifier.predict(input_df)[0]  # Assuming 'classifier' is your trained model

        return render_template('classify.html', prediction_text=f'Attendance classification: {prediction:.0f}')
    except Exception as e:
        return render_template('classify.html', prediction_text=f'Error: {str(e)}')


def preprocess_data(input_df):

    # Perform one-hot encoding on 'Size of Organization' and 'Type of Event'
    one_hot_cols = ['Org Size', 'Type']
    input_df = pd.get_dummies(input_df, columns=one_hot_cols)
    

    org_size = [
        'Org Size_Small (1-50 members)',
        'Org Size_Medium (50-200 members)',
        'Org Size_Large (200+ members)'
    ]

    # Create new features and preprocess data
    # input_df["Popular"] = input_df["Actual"] >= input_df["Expected"]
    # input_df["Popular"] = input_df["Popular"].astype(int)  # Convert boolean to integer
    for feature in org_size:
            if feature not in input_df.columns:
                # Add missing features as columns of zeros
                input_df[feature] = 0
    
    small_er = (input_df['Org Size_Small (1-50 members)'] * input_df['Expected'] / 25).fillna(0)
    
    mid_er = (input_df['Org Size_Medium (50-200 members)'] * input_df['Expected'] / 125).fillna(0)
    
    large_er = (input_df['Org Size_Large (200+ members)'] * input_df['Expected'] / 200).fillna(0)
    input_df["Expected Ratio"] = small_er + mid_er + large_er

    # Clip outliers
    input_df["Expected Ratio"] = input_df["Expected Ratio"].clip(lower=0.2, upper=4.0)

    # Remove unnecessary columns and prepare data for modeling
    input_df = input_df.drop(['Expected'], axis=1)
    print('mom', file=sys.stderr)
    
    return input_df

if __name__ == "__main__":
    app.run(debug=True)
