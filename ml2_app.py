import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import time  # For simulating real-time updates
from PIL import Image  # For image processing


# Define the models
def get_models(X):  # Pass the input features to the function
    """
    Define the regression models.

    Args:
        X (pd.DataFrame): The input features of the training data.

    Returns:
        dict: A dictionary of regression models.
    """
    # Get the number of rows in the training set
    n_rows_train = X.shape[0]
    print(f"Number Trees: {n_rows_train}")

    # Calculate the three potential values for n_estimators
    n_estimators_1 = int(np.ceil(n_rows_train * 0.5)) # Adjusted for reasonable values

    models = {
        "Linear Regression": LinearRegression(),
        f"Random Forest (n={n_estimators_1})": RandomForestRegressor(n_estimators=n_estimators_1, random_state=42),
        f"Gradient Boosting (n={n_estimators_1})": GradientBoostingRegressor(n_estimators=n_estimators_1, random_state=42),
        f"AdaBoost (n={n_estimators_1})": AdaBoostRegressor(n_estimators=n_estimators_1, random_state=42),
        f"Bagging (n={n_estimators_1})": BaggingRegressor(n_estimators=n_estimators_1, random_state=42),
    }
    return models


# Define image processing function
def process_image(df, image_path_cols, target_size=(100, 100)):  # Example target size
    """
    Reads, resizes, and converts an image to a 1D tensor (mean pixel value).

    Args:
        df (pd.DataFrame): The input DataFrame.
        image_path_cols (list): A list of image column paths names.
        target_size (tuple): The target size (width, height) for resizing.

    Returns:
        pd.DataFrame: DataFrame with new columns for mean pixel values and original path columns dropped.
    """
    df = df.copy()
    for image_path_col in image_path_cols:
        if image_path_col in df.columns:
            mean_pixels = []
            for filepath in df[image_path_col]:
                try:
                    img = Image.open(filepath).resize(target_size).convert('RGB')
                    img_array = np.array(img)
                    mean_pixels.append(np.mean(img_array))
                except Exception as e:
                    print(f"Error processing image {filepath}: {e}")
                    mean_pixels.append(np.nan)  # Handle errors by adding NaN
            df[f'mean_pixel_{image_path_col}'] = mean_pixels
            df = df.drop(columns=[image_path_col], errors='ignore') # Drop original, ignore if not exists
    return df

def extract_date_parts(df, date_cols):
    """
    Extracts date parts (year, month, day, day of week) from specified date columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_cols (list): A list of date column names.

    Returns:
        pd.DataFrame: A new DataFrame with the extracted date parts as separate columns,
                        and the original date columns dropped.
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame in place
    for date_col in date_cols:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce') # Handle potential parsing errors
                df[date_col + '_year'] = df[date_col].dt.year
                df[date_col + '_month'] = df[date_col].dt.month
                df[date_col + '_day'] = df[date_col].dt.day
                df[date_col + '_dayofweek'] = df[date_col].dt.dayofweek
                df = df.drop(columns=[date_col], errors='ignore') # Drop original, ignore if not exists
            except Exception as e:
                st.error(f"Error processing date column '{date_col}': {e}")
    return df


def train_and_evaluate(train_data, target_column):
    """
    Trains and evaluates multiple regression models using cross-validation.

    Args:
        train_data (pd.DataFrame): The training dataset.
        target_column (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing the trained models, cross-validation results,
               holdout set results, and the lists of processed features.
    """
    if target_column not in train_data.columns:
        st.error(f"Target column '{target_column}' not found in the training data.")
        return None, None, None, None, None, None

    # Separate features and target
    X = train_data.drop(target_column, axis=1)
    y = train_data[target_column]

    available_cols = X.columns.tolist()
    excluded_features = st.multiselect("Select features to exclude:", available_cols, default=[])
    print( excluded_features)
    binary_features = st.multiselect("Select binary features:", [col for col in available_cols if col not in excluded_features], default=[])
    date_features = st.multiselect("Select date features:", [col for col in available_cols if col not in excluded_features and col not in binary_features], default=[])
    image_path_col = st.multiselect("Select image path:", [col for col in available_cols], default=[])

    # Identify initial feature types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove excluded features
    numeric_features = [col for col in numeric_features if col not in excluded_features and col not in date_features and col not in image_path_col and col not in binary_features]
    categorical_features = [col for col in categorical_features if col not in excluded_features and col not in date_features and col not in image_path_col and col not in binary_features]

    # Type Casting
    for col in numeric_features:
        try:
            X[col] = pd.to_numeric(X[col])
        except ValueError:
            st.warning(f"Column {col} could not be fully converted to numeric.")
    for col in categorical_features:
        X[col] = X[col].astype('category')
    for col in binary_features:
        X[col] = X[col].astype('category') # Treat as categorical

    # Process date and image features
    X = extract_date_parts(X, date_features)
    X = process_image(X, image_path_col)

    # Update feature lists after processing
    #all_features = X.columns.tolist()
    #numeric_features_processed = X.select_dtypes(include=np.number).columns.tolist()
    #categorical_features_processed = X.select_dtypes(include=['object', 'category']).columns.tolist()
    #binary_features_processed = [col for col in all_features if col in binary_features] # Keep original selection

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bin', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), binary_features)
        ])

    # Split data for holdout set (before any preprocessing)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

    cv_results = {}
    trained_models = {}
    holdout_results = {} # To store holdout set performance

    # Get models after data processing
    models = get_models(X_train) # Pass X_train

    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),  # Use the preprocessor
                ('model', model)
            ])

            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Consistent CV
            cv_scores_rmse = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse = np.mean(np.sqrt(-cv_scores_rmse))  # Convert back to RMSE
            cv_scores_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
            cv_r2 = cv_scores_r2.mean()
            cv_results[name] = {'MeanCV_RMSE': cv_rmse, 'MeanCV_R2': cv_r2}

            # Train the model on the full training data (after CV)
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline

            st.write(f"Running Model: {name}")

            # Evaluate on the holdout set
            y_holdout_pred = pipeline.predict(X_holdout)
            holdout_rmse = root_mean_squared_error(y_holdout, y_holdout_pred)
            holdout_r2 = r2_score(y_holdout, y_holdout_pred)
            holdout_results[name] = {'RMSE': holdout_rmse, 'R2': holdout_r2}

        except Exception as e:
            st.error(f"Error training {name}: {e}")
            cv_results[name] ={'MeanCV_RMSE': np.nan, 'MeanCV_R2': np.nan}
            trained_models[name] = None
            holdout_results[name] = {'RMSE': np.nan, 'R2': np.nan}

    return trained_models, cv_results, holdout_results, numeric_features, categorical_features, binary_features, date_features, image_path_col

def predict_real_time(loaded_pipeline, test_data, numeric_features, categorical_features, binary_features, date_features, image_path_col):
    """
    Makes real-time predictions using a loaded trained pipeline.

    Args:
        loaded_pipeline (Pipeline): The loaded trained pipeline (including preprocessor and model).
        test_data (pd.DataFrame): The test dataset.
        numeric_features (list): List of numeric features used during training.
        categorical_features (list): List of categorical features used during training.
        binary_features (list): List of binary features used during training.
        date_features (list): List of date features used during training.
        image_path_col (list): List of image path columns used during training.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions. Returns an empty DataFrame on error.
    """
    predictions_df = pd.DataFrame()
    X_test = test_data.copy()

    # Process date features
    X_test = extract_date_parts(X_test, date_features)

    # Process image features
    X_test = process_image(X_test, image_path_col)

    # Ensure categorical columns are of 'category' dtype
    for col in categorical_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
    for col in binary_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    # Select only the features that were used during training
    # This is crucial to match the columns expected by the pipeline
    training_columns = (
        numeric_features +
        [f"{col}_year" for col in date_features if col in X_test.columns] +
        [f"{col}_month" for col in date_features if col in X_test.columns] +
        [f"{col}_day" for col in date_features if col in X_test.columns] +
        [f"{col}_dayofweek" for col in date_features if col in X_test.columns] +
        [f"mean_pixel_{col}" for col in image_path_col if col in X_test.columns] +
        [col for col in categorical_features if col in X_test.columns] +
        [col for col in binary_features if col in X_test.columns]
    )

    # Add a try-except block for potential missing columns
    try:
        X_test = X_test[training_columns]
    except KeyError as e:
        st.error(f"Error: Test data is missing the following columns required by the trained model: {e}")
        return pd.DataFrame()

    try:
        # Make predictions using the loaded pipeline
        y_pred = loaded_pipeline.predict(X_test)
        predictions_df['prediction'] = y_pred  # Store predictions in the DataFrame
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    return predictions_df

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Regression Model Trainer and Real-time Predictor")

    # File upload
    train_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    test_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

    target_column = st.text_input("Enter the name of the target variable column:", "target")

    if train_file is not None:
        try:
            train_data = pd.read_csv(train_file)
            st.write("Training Data:")
            st.dataframe(train_data.head())

            if st.subheader("Run Training and Evaluate"):
                trained_models, cv_results, holdout_results, numeric_features, categorical_features, binary_features, date_features, image_path_col = train_and_evaluate(train_data, target_column)

                if cv_results and holdout_results:
                    st.subheader("Cross-Validation Results (RMSE):")
                    for name, scores in cv_results.items():
                        st.write(f"{name}: MeanCV RMSE={scores['MeanCV_RMSE']:.4f}, MeanCV R2={scores['MeanCV_R2']:.4f}")

                    st.subheader("Holdout Set Results:")
                    for name, scores in holdout_results.items():
                        st.write(f"{name}: RMSE={scores['RMSE']:.4f}, R2={scores['R2']:.4f}")

                    # Find the best model based on Holdout R2
                    best_model_name_r2 = max(holdout_results, key=lambda x: holdout_results[x]['R2'])
                    st.subheader(f"Best Model (Holdout R2): {best_model_name_r2}")
                    st.write(f"R2: {holdout_results[best_model_name_r2]['R2']:.4f}, RMSE: {holdout_results[best_model_name_r2]['RMSE']:.4f}")
                    best_model_r2 = trained_models[best_model_name_r2]

                    # Find the best model based on Holdout RMSE
                    best_model_name_rmse = min(holdout_results, key=lambda x: holdout_results[x]['RMSE'])
                    st.subheader(f"Best Model (Holdout RMSE): {best_model_name_rmse}")
                    st.write(f"RMSE: {holdout_results[best_model_name_rmse]['RMSE']:.4f}, R2: {holdout_results[best_model_name_rmse]['R2']:.4f}")
                    best_model_rmse = trained_models[best_model_name_rmse]

                    # Allow user to select which best model to use for prediction.
                    best_model_selection = st.radio(
                        "Select best model to use for prediction:",
                        [f"Best by R2 ({best_model_name_r2})", f"Best by RMSE ({best_model_name_rmse})"],
                        index=0,  # Default to the first model in the list
                    )

                    selected_model = best_model_r2 if "R2" in best_model_selection else best_model_rmse
                    selected_model_name = best_model_name_r2 if "R2" in best_model_selection else best_model_name_rmse

                    # Save the best model (the entire pipeline)
                    if st.checkbox("Save Best Model"):
                        try:
                            joblib.dump(selected_model, "best_model.pkl")
                            st.success(f"Best model ({selected_model_name}) saved as best_model.pkl")
                            best_model_file = "best_model.pkl"
                        except Exception as e:
                            st.error(f"Error saving model: {e}")

                    # Section Real Time Predictions
                    st.subheader("Real-time Predictions")
                    if test_file is not None:
                        try:
                            test_data = pd.read_csv(test_file)
                            st.write("Test Data:")
                            st.dataframe(test_data.head())
                        except Exception as e:
                            st.error(f"Error reading test data: {e}")
                            test_data = None #set test data to none
                        if st.button("Start Real-time Predictions"):
                            try:
                                loaded_pipeline = joblib.load(best_model_file )
                                predictions = predict_real_time(loaded_pipeline, test_data, numeric_features, categorical_features, binary_features, date_features, image_path_col)
                                if not predictions.empty:
                                    st.write("Predictions for all test data:")
                                    st.dataframe(predictions)
                                else:
                                    st.warning("No predictions could be made. Please check your data and model.")
                            except FileNotFoundError:
                                st.error("Best model file not found. Please train and save the model first.")
                            except Exception as e:
                                st.error(f"Error loading the model: {e}")
                    else:
                        st.warning("Please upload test data to make predictions.")
                else:
                    st.warning("No models were successfully trained.")
        except Exception as e:
            st.error(f"Error reading training data: {e}")

if __name__ == "__main__":
    main()