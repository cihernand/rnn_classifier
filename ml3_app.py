import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
from PIL import Image
import os
import datetime
# --- Page Configuration ---
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# --- Function to generate a timestamped filename ---
def get_timestamped_filename(base_name="model", extension=".pkl"):
    """Generates a filename with the current date and time."""
    now = datetime.datetime.now()
    # Format: YYYYMMDD_HHMMSS (e.g., 20250520_184208)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


# --- Select Features ---
@st.cache_data
def select_features(X_train_processed, X_test_processed,
                             importance_threshold=0.010, _model=None, prefix_to_drop='unimportant_',
                             ):
    """
    Selects important features based on Random Forest feature importances,
    evaluates a model (default: RandomForestRegressor) with cross-validation,
    and reports performance on the test set.

    Args:
        X_train_processed (pd.DataFrame): Preprocessed training data.
        X_test_processed (pd.DataFrame): Preprocessed testing data.
        importance_threshold (float, optional): Threshold for feature importance.
            Defaults to 0.015.
        model: A trained model with a 'feature_importances_' attribute (e.g., RandomForestRegressor).
        prefix_to_drop (str, optional): Prefix of columns to drop.
            Defaults to 'unimportant_'. 

    Returns:

        - pd.DataFrame: DataFrame with selected features for training.
        - pd.DataFrame: DataFrame with selected features for testing.
    
    """
    try:
        
        feature_importances = pd.Series(_model.feature_importances_,
                                        index=X_train_processed.columns)

        # Select important features
        important_features = feature_importances[
            feature_importances >= importance_threshold].index.tolist()

        # Create new DataFrames with only the important features
        X_train_selected = X_train_processed[important_features]
        X_test_selected = X_test_processed[important_features]

        # Drop columns with the specified prefix
        columns_to_drop = [
            col for col in X_train_selected.columns
            if col.startswith(prefix_to_drop) or col.startswith('image_')
        ]
        X_train_final = X_train_selected.drop(columns=columns_to_drop,
                                              errors='ignore')
        X_test_final = X_test_selected.drop(columns=columns_to_drop,
                                            errors='ignore')

        selected_features = X_train_final.columns.to_list() 

        return X_train_final, X_test_final, selected_features

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

# --- Function to Train and Evaluate ML model  ---
# This caches the model so it's not re-trained on every rerun with the same params
@st.cache_data
def evaluate_model(X_train, y_train, X_test, y_test, _model=None):
    """
    Evaluates a regression model using cross-validation on the training set
    and reports performance on the testing set.

    Args:
        X_train (pd.DataFrame):  training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame):  testing features.
        y_test (pd.Series): Testing target variable.
        _model: A pre-initialized regression model.  Defaults to a RandomForestRegressor.
               If None, a RandomForestRegressor is initialized with
               random_state=42 and n_estimators=200.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            'cv_rmse': Cross-validation RMSE on the training set.
            'cv_r2': Cross-validation R-squared on the training set.
            'test_rmse': RMSE on the testing set.
            'test_r2': R-squared on the testing set.
            'test_errors_standardized': Standardized errors on the testing set.
            Returns None if an error occurs.
    """
    try:
        if _model is None:
            _model = RandomForestRegressor(random_state=42,
                                          n_estimators=200)  # Default model

        # Model Training and Cross-validation
        cv = KFold(n_splits=5, shuffle=True,
                   random_state=42)  # Consistent CV
        cv_scores_rmse = cross_val_score(_model, X_train, y_train, cv=cv,
                                         scoring='neg_mean_squared_error')
        cv_rmse = np.mean(
            np.sqrt(-cv_scores_rmse))  # Convert negative MSE to positive RMSE
        cv_scores_r2 = cross_val_score(_model, X_train, y_train, cv=cv,
                                       scoring='r2')
        cv_r2 = cv_scores_r2.mean()

        # Testing Set Evaluation
        test_model = _model.fit(X_test, y_test) # Fit on the test set
        y_pred_test = test_model.predict(X_test)
        test_rmse = root_mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        errors_test = y_test - y_pred_test
        errors_test_standard = StandardScaler().fit_transform(
            errors_test.values.reshape(-1, 1))

        return _model, cv_rmse, cv_r2, test_rmse, test_r2, errors_test_standard, y_pred_test
    
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return None

# --- Function to do OneHotEncoding and Standard Scaling ---

def preprocess_data(X, categorical_features, numeric_features):
    """
    Preprocesses data by encoding categorical features and scaling numeric features.

    Args:
        X (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical feature names.
        numeric_features (list): List of numeric feature names.

    Returns:
        pd.DataFrame: The processed  DataFrame.
               Returns None if an error occurs during processing.
    """
    try:


        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Fit and transform the encoder on the  data's categorical columns
        encoder.fit(X[categorical_features])
        X_encoded = encoder.transform(X[categorical_features])
  
        # Get the feature names (categories) generated by the encoder
        feature_names = encoder.get_feature_names_out(categorical_features)
        

        # Create new DataFrames with the encoded features, handling potential index issues
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)

        # Drop the original categorical columns and concatenate the encoded ones
        X_processed = pd.concat([X.drop(categorical_features, axis=1, errors='ignore'), X_encoded_df], axis=1)

        # Initialize and fit scaler on the data
        scaler = StandardScaler()
        X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
        print(X_processed.head())

        return X_processed

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

# --- Function to Estimate Correlations ---
def get_significant_correlations(df, target_variable=None, alpha=0.05):
    """
    Identifies variables in a Pandas DataFrame that have statistically significant
    correlations, optionally with respect to a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_variable (str, optional): The name of the target variable. If provided,
            the function calculates correlations only between the target variable
            and other variables. Defaults to None, in which case it calculates
            correlations between all pairs of variables.
        alpha (float, optional): The significance level for determining
            statistical significance. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame of significant correlations
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    cols = numeric_df.columns
    correlations = []

    if target_variable:
        if target_variable not in cols:
            return pd.DataFrame()
        for var in cols:
            if var != target_variable:
                correlation, p_value = pearsonr(numeric_df[target_variable], numeric_df[var])
                if p_value < alpha:
                    correlations.append({'Variable1': target_variable, 'Variable2': var, 'Correlation': correlation, 'P-value': p_value})
    else:
        for i, var1 in enumerate(cols):
            for j, var2 in enumerate(cols[i+1:]):
                correlation, p_value = pearsonr(numeric_df[var1], numeric_df[var2])
                if p_value < alpha:
                    correlations.append({'Variable1': var1, 'Variable2': var2, 'Correlation': correlation, 'P-value': p_value})
    
    correlations_df = pd.DataFrame(correlations)
    
    if not correlations_df.empty:
        correlations_df = correlations_df.sort_values(by='Correlation', ascending=False)  # Sort by correlation
 
    else:
        st.info("No significant correlations found.")

    return correlations_df

# --- Function to Get Numerical and Categorical Vars ---

def get_numeric_categorical_variables(df):
    """
    Gets numeric and categorical variables from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains the names of the numeric variables.
            - The second list contains the names of the categorical variables.
            - Returns empty lists if the input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a Pandas DataFrame. Returning empty lists.")
        return [], []

    numeric_variables = list(df.select_dtypes(include=[np.number]).columns)
    categorical_variables = list(df.select_dtypes(include=['object', 'category']).columns)

    return numeric_variables, categorical_variables


# --- Function to Drop Variables ---
def drop_variables(df, variables_to_drop):
    """
    Drops a list of variables from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables_to_drop (list): A list of variable names (strings) to drop from the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the specified variables dropped.
                      Returns the original DataFrame if variables_to_drop is empty
                      or if a variable is not found in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(variables_to_drop, list):
        raise TypeError("variables_to_drop must be a list")

    if not variables_to_drop:  # Check for empty list
        return df

    # Check if all variables_to_drop are in df
    missing_variables = [var for var in variables_to_drop if var not in df.columns]
    if missing_variables:
        print(f"Warning: The following variables were not found in the DataFrame: {missing_variables}")
        # Drop only the variables that are found
        variables_to_drop = [var for var in variables_to_drop if var in df.columns]
        if not variables_to_drop: # if the intersection is empty
            return df

    try:
        df = df.drop(columns=variables_to_drop)
        return df
    except KeyError as e:
        print(f"KeyError: {e}.  Returning original DataFrame.")
        return df




# --- Function to process date column and clean column headers ---
def process_and_clean_data(df):
    """
    Processes a pandas DataFrame to clean column names, extract date features,
    set 'date' as the index, and sort the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be processed.
            It is assumed to contain a column named 'date' with datetime values.

    Returns:
        pd.DataFrame: The processed DataFrame with cleaned column names,
            added date feature columns ('date_year', 'date_month',
            'date_day', 'date_dayofweek'), 'date' as the index,
            and sorted by the index.  Returns the original DataFrame
            if 'date' column does not exist or is not a datetime.
    """
    # Check if the input is a DataFrame and contains the 'date' column
    if not isinstance(df, pd.DataFrame) or 'date' not in df.columns:
        return df

    # Helper function to clean column names
    def clean_header(col):
        return str(col).strip().replace(' ', '_')

    df.columns = [clean_header(col) for col in df.columns]

    try:
        # Extract date features
        date_col = pd.to_datetime(df['date'])  # Convert 'date' to datetime
        df['date_year'] = date_col.dt.year
        df['date_month'] = date_col.dt.month_name()
        df['date_day'] = date_col.dt.day
        df['date_dayofweek'] = date_col.dt.day_name()

        # Sort and set 'date' as index
        df = df.sort_values(by='date')
        df = df.drop('date', axis=1)
        return df

    except (AttributeError, KeyError):
        return df

# --- Function to Upload CSV Files ---  
def upload_csv(title):
    """
    Uploads a CSV file and returns the DataFrame.

    Args:
        title (str): The title of the file upload widget.

    Returns:
        pandas.DataFrame: The uploaded DataFrame, or None if no file is uploaded.
    """
    uploaded_file = st.sidebar.file_uploader(title, type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
            return None
    return None

#Start Figures
st.session_state.plotly_figures = []

# --- Main Content ---
st.title("ML model Data Analysis Dashboard")

# --- Data Loading ---
df_training = upload_csv("First Step. Upload Training Set CSV")

# --- Header 1: Feature Preprocessing ---
st.header("Preprocessing features")
col1, col2, col3 = st.columns(3)

if df_training is not None:
    try:
        # Process Input Columns
        df_training = process_and_clean_data(df_training)
        st.write ("Column date is processed to get date_year, date_month, date_day and date_dayofweek")

        # Target variable selection
        target_variable = st.sidebar.selectbox("Select the target variable:", df_training.columns)
        # Exclude variables 
        excluded_variables = st.sidebar.selectbox("Select the variables to exclude:", df_training.columns)
        excluded_variables =[excluded_variables]

        # Drop columns
        df_training = drop_variables (df_training, excluded_variables)

        # Estimate Correlation with target vars
        df_cor_all = get_significant_correlations(df_training)
        var1 = df_cor_all.iloc[0,0]
        var2 = df_cor_all.iloc[0,1]
        
        with col1:
            col1.subheader("Correlation between Features")
            st.dataframe(df_cor_all)
         
        with col2:
            # Plot Variables with high correlation
            col2.subheader("Correlated Features")
            fig_scatter, ax_scatter = plt.subplots()
            plt.scatter(df_training[var1], df_training[var2])
            plt.xlabel(var1)
            plt.ylabel(var2)
            st.pyplot(fig_scatter)
            st.write(f"Feature: {var1} is redundant and will be removed")

        with col3:
            # Plot Variables with high correlation                      
            col3.subheader("Target Variable")
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df_training[target_variable], kde=True, ax=ax_hist)  # Use the target variable
            st.pyplot(fig_hist)
           
        df_training = drop_variables(df_training, [var1])


        
        # Create X and Y Objects
        y = df_training[target_variable]   # Target
        X = drop_variables(df_training, [target_variable])

        # Get numerical and categorical variables
        num_vars, cat_vars = get_numeric_categorical_variables(X)
      

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

        X_train_processed = preprocess_data(X_train,cat_vars,num_vars)
        X_test_processed = preprocess_data(X_test,cat_vars,num_vars)  

        st.subheader("One Hot Encoding and Standard Transformation Training Set")  
        st.dataframe(X_train_processed.describe())              

    except KeyError as e:
        st.error(f"KeyError: {e}. Adjust the column names as needed.")

else:
    st.write("No training data uploaded.")

# --- Header 1: Model Results ---
st.header("Running Model with all features")


if df_training is not None:
    try:          

        # Train a Linear Regression model (you can replace with your best model)
        rf_model = RandomForestRegressor(random_state=42,n_estimators= 200)
                    
        model_v1, cv_rmse, cv_r2, test_rmse, test_r2, errors_test_standard,y_pred_test  = evaluate_model(X_train_processed,
                                                                                         y_train,
                                                                                         X_test_processed,
                                                                                         y_test, _model=rf_model)
        
        # The model object is now in this variable:
        my_trained_model_v1 = model_v1

        # --- Streamlit UI for saving ---
        st.write(f"The model {my_trained_model_v1} will be stored into the local directory.")

        model_base_name_v1 = st.text_input(
        "Please write a name for your model file:",
        value="my_model_all_features",
        key = "key1"
        )

        if st.button("Save First Model"):

            if not model_base_name_v1.strip():
                st.warning("Please enter a base name for the model file.")
            else:
                # Generate the unique filename with timestamp
                filename_v1 = get_timestamped_filename(model_base_name_v1.strip())
                
                # When you only provide a filename, it saves to the current directory.
                full_path_to_save = filename_v1

            # Save the model object using pickle
            with open(full_path_to_save, 'wb') as file: # 'wb' for write binary
                pickle.dump(my_trained_model_v1, file)
                file.close()            
                st.success(f"Model successfully saved to: `{full_path_to_save}`")

        try:

            st.subheader("Model Evaluation")

            col1, col2, col3 = st.columns(3)

            with col1:
                col1.subheader("Model")
                st.write(f"Params: {model_v1.get_params}")

        
            with col2:
                col2.subheader("CV Training Results")
                st.write(f"CV RMSE: {cv_rmse:.2f}")
                st.write(f"CV R2: {cv_r2:.2f}")
                st.write(f"Rows in Set: {X_train_processed.shape[0]}")

            with col3:
                col3.subheader("Testing Results")
                st.write(f"Test RMSE: {test_rmse:.2f}")
                st.write(f"Test R2: {test_r2:.2f}")
                st.write(f"Rows in Set: {X_test_processed.shape[0]}")



            with col1:
                col1.subheader("Test Obs vs Pred")
                fig_scatter, ax_scatter2 = plt.subplots()
                plt.scatter(y_test, y_pred_test)
                plt.xlabel("y_observed")
                plt.ylabel("y_predicted")
                st.pyplot(fig_scatter)

            with col2:
                col2.subheader("Test Standard Errors")
                fig_hist2, ax_hist = plt.subplots()
                sns.histplot(errors_test_standard)
                plt.xlabel ("standard errors \n y_true - y_pred")
                plt.ylabel ("Frequency")
                st.pyplot(fig_hist2)

            st.subheader("Feature Importances")
            # Get feature importances, sort them, and create a DataFrame
            sorted_indices = np.argsort(my_trained_model_v1.feature_importances_)[::-1]
            sorted_features = X_test_processed.columns.values[sorted_indices]
            sorted_importances = my_trained_model_v1.feature_importances_[sorted_indices]
            sorted_importances_df = pd.DataFrame(sorted_importances, index=sorted_features, columns=['Importance'])


            # Use Streamlit to create a bar chart
            st.bar_chart(sorted_importances_df)

            # Add a caption
            st.caption(f"Bar chart of feature importances from the model name: {filename_v1}")
        

        except Exception as e:
            st.error(f"An error in plots display: {e}")
    
                
    except KeyError as e:
        st.error(f"KeyError: {e}. Adjust the column names as needed.")
    except Exception as e:
        st.error(f"An error occurred during model training/evaluation: {e}")
else:
    st.write("No training data uploaded.")

# --- Header 2: Model Results with Selected Features ---
st.header("Running Model with Selected Features")


if model_v1 is not None:
    try:
        X_train_sel, X_test_sel, selected_features = select_features(X_train_processed,X_test_processed,importance_threshold=0.10, _model = model_v1)
        model_v2, cv_rmse, cv_r2, test_rmse, test_r2, errors_test_standard,y_pred_test  = evaluate_model(X_train_sel,
                                                                                         y_train,
                                                                                         X_test_sel,
                                                                                         y_test, _model=rf_model)


        # The model object is now in this variable:
        my_trained_model_v2 = model_v2

        # --- Streamlit UI for saving ---
        st.write(f"The model {my_trained_model_v2} will be stored into the local directory.")

        model_base_name_v2 = st.text_input(
        "Please write a name for your model file:",
        value="my_model_selected_features",
        key = "key2"
        )

        if st.button("Save Second Model"):

            if not model_base_name_v2.strip():
                st.warning("Please enter a base name for the model file.")
            else:
                # Generate the unique filename with timestamp
                filename_v2 = get_timestamped_filename(model_base_name_v2.strip())
                
                # When you only provide a filename, it saves to the current directory.
                full_path_to_save = filename_v2

            # Save the model object using pickle
            with open(full_path_to_save, 'wb') as file: # 'wb' for write binary
                pickle.dump(my_trained_model_v2, file)
                file.close()            
                st.success(f"Model successfully saved to: `{full_path_to_save}`")


        try:

            st.subheader("Model Evaluation")
            st.write(f"Selected Features for new model \n:{selected_features}")

            col5, col6, col7 = st.columns(3)

            with col5:
                col5.subheader("Model")
                st.write(f"Params: {model_v2.get_params}")

        
            with col6:
                col6.subheader("CV Training Results")
                st.write(f"CV RMSE: {cv_rmse:.2f}")
                st.write(f"CV R2: {cv_r2:.2f}")
                st.write(f"Rows in Set: {X_train_processed.shape[0]}")

            with col7:
                col7.subheader("Testing Results")
                st.write(f"Test RMSE: {test_rmse:.2f}")
                st.write(f"Test R2: {test_r2:.2f}")
                st.write(f"Rows in Set: {X_test_sel.shape[0]}")

            with col5:
                col5.subheader("Test Obs vs Pred")
                fig_scatter, ax_scatter2 = plt.subplots()
                plt.scatter(y_test, y_pred_test)
                plt.xlabel("y_observed")
                plt.ylabel("y_predicted")
                st.pyplot(fig_scatter)

            with col6:
                col6.subheader("Test Standard Errors")
                fig_hist2, ax_hist = plt.subplots()
                sns.histplot(errors_test_standard)
                plt.xlabel ("standard errors \n y_true - y_pred")
                plt.ylabel ("Frequency")
                st.pyplot(fig_hist2)


            st.subheader("Feature Importances")
            # Get feature importances, sort them, and create a DataFrame
            sorted_indices = np.argsort(my_trained_model_v2.feature_importances_)[::-1]
            sorted_features = X_test_sel.columns.values[sorted_indices]
            sorted_importances = my_trained_model_v2.feature_importances_[sorted_indices]
            sorted_importances_df = pd.DataFrame(sorted_importances, index=sorted_features, columns=['Importance'])


            # Use Streamlit to create a bar chart
            st.bar_chart(sorted_importances_df)

            # Add a caption
            st.caption(f"Bar chart of feature importances from the model name: {filename_v2}")
            
        except Exception as e:
            st.error(f"An error occurred while saving the model: {e}")


    except KeyError as e:
        st.error(f"KeyError: {e}.  Ensure the necessary columns exist in the training data.")
    except Exception as e:
        st.error(f"An error occurred during model training/evaluation: {e}")
else:
    st.write("No training data uploaded.")

# --- Header 3: Predicted Values ---
st.header("Predicted New Data Values")

if df_training is not None:
    try:
        #  Load new Data
        new_data_input = upload_csv("Secod Step. Upload New Data for Predictions") # added file uploader
        if new_data_input is not None:

            # Initial Preprocessing
            new_data = process_and_clean_data(new_data_input)

            # Target variable selection
            new_excluded_variables  = st.sidebar.selectbox("Select the variables to exclude::", new_data.columns)
            # Exclude variables 
            new_path_img_variable = st.sidebar.selectbox("Select column with path to images:", new_data.columns)

            # Ask user if process images
            st.write("The model doesn't predict image class, do you  want to see the images from the uploaded newdata :")
            user_choice = st.radio("Select your answer:",("Yes", "No"),index=None, key="yes_no_selector")

            #Load Images
            if user_choice == "Yes":

                st.write("After image display, the predictions will be estimated.")                    

                # --- Create 5 columns ---
                # This allows us to cycle through them
                columns = st.columns(5) # This creates exactly 5 columns of equal width

                # --- Iterate through images and place them in columns ---
                for i, image_path in enumerate(new_data[new_path_img_variable]):
                    # Determine which column to place the current image in
                    # The modulo operator (%) cycles through the columns (0, 1, 2, 3, 4, 0, 1, ...)
                    col = columns[i % 5]

                    with col: # Use the 'with' syntax to place elements into the specific column
                        try:
                            # Open image with Pillow for consistency and to handle potential issues
                            img = Image.open(image_path)
                            resized_img = img.resize((100, 100))
                            st.image(resized_img, caption="", width=100)
                            

                        except Exception as e:
                            st.error(f"Could not load image {os.path.basename(image_path)}: {e}")             
                
            elif user_choice == "No":

                st.write("The predictions will be estimated.")
            
            else:

                st.warning("Please select an answer.")

            if user_choice is not None:


                # Drop columns
                new_data = drop_variables (new_data, [new_excluded_variables])

                # Get numerical and categorical variables
                num_vars_nd, cat_vars_nd = get_numeric_categorical_variables(new_data)

                new_data_processed = preprocess_data(new_data,cat_vars_nd,num_vars_nd)

                st.subheader("New Data Set One Hot Encoding and Standard Transformation ")  
                st.dataframe(new_data_processed.describe())  

                #Preprocess
                X_pred = new_data_processed[selected_features] # Use the features selected for the "best" model


                
                y_pred = model_v2.predict(X_pred)
                predictions_df = pd.DataFrame({'target': y_pred})

                # Add other columns from the new data input to the output DataFrame
                predictions_and_vars_df = pd.concat([predictions_df, new_data_input], axis=1)

                # Create Columns to display predictions
                col9, col10  = st.columns(2)

                with col9:
                    col9.subheader("Predicted target")
                    st.dataframe(predictions_df.head())
                with col10:
                    col10.subheader("Predicted target Distribution")
                    fig_pred_dist, ax_pred_dist = plt.subplots()
                    sns.histplot(predictions_df['target'], kde=True, ax=ax_pred_dist)
                    st.pyplot(fig_pred_dist)

                st.subheader("New Data Set with predictions")  
                st.dataframe(predictions_and_vars_df.head())

        else:
            st.write("Please on the left menu, upload new data to get predictions.")

    except KeyError as e:
        st.error(f"KeyError: {e}.  Ensure the uploaded data contains the columns: {selected_features}.  These are the features the model was trained on.")
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.write("No new data uploaded.")
