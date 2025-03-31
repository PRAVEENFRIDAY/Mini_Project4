import base64
import os
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Use @st.cache_resource for caching heavy objects like models
@st.cache_resource
def load_data():
    try:
        # Print the current working directory
        current_dir = os.getcwd()
        print("Current working directory:", current_dir)

        # Absolute path to the directory containing the pickle files
        pickle_dir = r'D:/GuviCoruseDoc/Data_Science/Mini_Project_4/env/Scripts'

        # List of pickle files
        pickle_files = [
            'rf_regressor.pkl', 'recomClassifier.pkl', 'label_encoder.pkl', 'rclabel_encoder.pkl',
            'user_mat_sim.pkl', 'user_item_matrix.pkl', 'user_recom.pkl', 'df.pkl', 'mode_df.pkl', 'type_df.pkl',
            'X_test.pkl', 'y_test.pkl', 'X_test_clf.pkl', 'y_test_clf.pkl'
        ]

        # Check if each file exists
        for file in pickle_files:
            file_path = os.path.join(pickle_dir, file)
            print(f"Checking file: {file_path}")
            if not os.path.exists(file_path):
                st.error(f"Error: {file} is missing at {file_path}.")
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                print(f"Found file: {file_path}")

        # Load each pickle file individually
        with open(os.path.join(pickle_dir, 'rf_regressor.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(pickle_dir, 'recomClassifier.pkl'), 'rb') as f:
            rc = pickle.load(f)
        with open(os.path.join(pickle_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(pickle_dir, 'rclabel_encoder.pkl'), 'rb') as f:
            rclabel_encoder = pickle.load(f)
        with open(os.path.join(pickle_dir, 'user_mat_sim.pkl'), 'rb') as f:
            user_mat_sim = pickle.load(f)
        with open(os.path.join(pickle_dir, 'user_item_matrix.pkl'), 'rb') as f:
            user_item_matrix = pickle.load(f)
        with open(os.path.join(pickle_dir, 'user_recom.pkl'), 'rb') as f:
            user_recom = pickle.load(f)
        with open(os.path.join(pickle_dir, 'df.pkl'), 'rb') as f:
            df = pickle.load(f)
        with open(os.path.join(pickle_dir, 'mode_df.pkl'), 'rb') as f:
            mode_df = pickle.load(f)
        with open(os.path.join(pickle_dir, 'type_df.pkl'), 'rb') as f:
            type_df = pickle.load(f)
        with open(os.path.join(pickle_dir, 'X_test.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        with open(os.path.join(pickle_dir, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        with open(os.path.join(pickle_dir, 'X_test_clf.pkl'), 'rb') as f:
            X_test_clf = pickle.load(f)
        with open(os.path.join(pickle_dir, 'y_test_clf.pkl'), 'rb') as f:
            y_test_clf = pickle.load(f)

        return (model, rc, label_encoder, rclabel_encoder, user_mat_sim, 
                user_item_matrix, user_recom, df, mode_df, type_df,
                X_test, y_test, X_test_clf, y_test_clf)
    except EOFError:
        st.error("Error: One or more pickle files are corrupted. Please recreate them.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    except FileNotFoundError:
        st.error("Error: One or more pickle files are missing.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

# Load all data including test data
(model, rc, label_encoder, rclabel_encoder, user_mat_sim, 
 user_item_matrix, user_recom, df, mode_df, type_df,
 X_test, y_test, X_test_clf, y_test_clf) = load_data()

# Clear cache button
if st.button('Clear Cache'):
    st.cache_resource.clear()
    st.experimental_rerun()

# Define features for regression and classification
features = ['UserId', 'CityName', 'Attraction', 'VisitMode', 'AttractionType']
RegClassfeatures = ['CityName', 'Rating', 'Attraction']

if model is None:
    st.stop()

def evaluate_regression(model, X_test, y_test):
    """Evaluate regression model and return metrics"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'R-squared Score': r2
    }
    
    # Create a residual plot
    residuals = y_test - y_pred
    fig = px.scatter(x=y_pred, y=residuals, 
                     labels={'x': 'Predicted Values', 'y': 'Residuals'},
                     title='Residual Plot')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    return metrics, fig

def evaluate_classification(model, X_test, y_test, label_encoder=None):
    """Evaluate classification model and return metrics"""
    try:
        y_pred = model.predict(X_test)

        # Check for consistent sample sizes
        if len(y_test) != len(y_pred):
            st.error(f"Error: Inconsistent sample sizes. y_test has {len(y_test)} samples, y_pred has {len(y_pred)} samples.")
            return None, None

        # Compute metrics before decoding (if encoded)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        # Create confusion matrix with encoded labels if needed
        cm = confusion_matrix(y_test, y_pred)
        
        # Try to decode labels for display if encoder is available
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            try:
                y_test_labels = label_encoder.inverse_transform(y_test)
                y_pred_labels = label_encoder.inverse_transform(y_pred)
                labels = label_encoder.classes_
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=labels, y=labels,
                               title='Confusion Matrix')
            except ValueError as e:
                st.warning(f"Could not decode labels for confusion matrix: {str(e)}")
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               title='Confusion Matrix (Encoded Labels)')
        else:
            fig = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           title='Confusion Matrix (Encoded Labels)')

        return metrics, fig

    except Exception as e:
        st.error(f"Error in classification evaluation: {str(e)}")
        return None, None


def add_bg_from_local(image_file):
    # Absolute path to the image file
    image_path = os.path.join(r'D:/GuviCoruseDoc/Data_Science/Mini_Project_4/env/Scripts', image_file)
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Error: {image_file} is missing at {image_path}.")

add_bg_from_local('Classified.jpg')

# Recommendation functions
def collabfill(user_id, user_mat_sim, user_item_matrix, user_recom):
    num_users = user_mat_sim.shape[0]

    if user_id < 1 or user_id > num_users:
        st.error(f"Error: User ID {user_id} is out of range. Valid range: 1 to {num_users}")
        return pd.DataFrame()

    simuser = user_mat_sim[user_id - 1]
    sim_user_ids = np.argsort(simuser)[::-1][1:6]
    sim_user_rating = user_item_matrix.iloc[sim_user_ids].mean(axis=0)
    rec_dest_id = sim_user_rating.sort_values(ascending=False).head(5).index
    rec = user_recom[user_recom['AttractionId'].isin(rec_dest_id)][['Attraction', 'VisitMode', 'Rating']].drop_duplicates().head(5)

    return rec

def recom_dest(user_input, model, label_encoder, features, df):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    pred_rate = model.predict(input_df)[0]
    return int(pred_rate)

def recomClassifier_dest(user_input, model, label_encoder, features, df):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            if user_input[i] in label_encoder[i].classes_:
                encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
            else:
                st.error(f"Error: Unknown value '{user_input[i]}' for feature '{i}'")
                return None
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    input_df = input_df[features]  # Reorder columns to match RegClassfeatures
    print("Input DataFrame Columns:", input_df.columns)
    pred_rate = model.predict(input_df)[0]
    return pred_rate

# Streamlit App
st.title("Tourism Prediction Analysis")

def recommendation():
    st.header("Personalized Attraction Suggestions")
  
    with st.form(key='recommendation_form'):
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        city = st.text_input("Enter City Name (Optional)")
        attraction = st.text_input("Enter Attraction Name (Optional)")
        mode = st.selectbox("Select Visit Mode (Optional)", mode_df['VisitMode'].unique())
        attraction_type = st.selectbox("Select Attraction Type (Optional)", type_df['AttractionType'].unique())
        submit_button = st.form_submit_button(label='Recommend')

    if submit_button:
        recom_data = collabfill(user_id, user_mat_sim, user_item_matrix, user_recom)        
        if not recom_data.empty:
            st.subheader("Recommended Destinations:")
            st.dataframe(recom_data)            
        else:
            st.warning("No recommendations found for this user.")

def regression():
    st.header("Predicting Attraction Ratings")
    
    # Add evaluation section
    with st.expander("Model Evaluation Metrics", expanded=False):
        if X_test is not None and y_test is not None:
            reg_metrics, reg_plot = evaluate_regression(model, X_test, y_test)
            
            st.subheader("Regression Metrics")
            col1, col2 = st.columns(2)
            with col1:
                for name, value in reg_metrics.items():
                    st.metric(label=name, value=f"{value:.4f}")
            
            st.subheader("Residual Plot")
            st.plotly_chart(reg_plot, use_container_width=True)
        else:
            st.warning("Test data not available for evaluation")
    
    st.write("Predict the rating a user might give to a tourist attraction based on historical data.")
    with st.form(key='regression_form'):
        user_id_reg = st.number_input("Enter User ID", min_value=1, step=1)
        city_reg = st.text_input("Enter City Name")
        attraction_reg = st.text_input("Enter Attraction Name")
        mode_reg = st.selectbox("Select Visit Mode", mode_df['VisitMode'].unique(), key='reg_mode')
        attraction_type_reg = st.selectbox("Select Attraction Type", type_df['AttractionType'].unique(), key='reg_type')
        continent_reg = st.text_input("Enter Continent (Optional)")
        region_reg = st.text_input("Enter Region (Optional)")
        country_reg = st.text_input("Enter Country (Optional)")
        year_reg = st.number_input("Enter Year (Optional)", min_value=1900, max_value=2100, value=2023, step=1)
        month_reg = st.number_input("Enter Month (Optional)", min_value=1, max_value=12, value=1, step=1)

        submit_button_reg = st.form_submit_button(label='Predict Rating')

    if submit_button_reg:
        user_input = {
            'UserId': user_id_reg,
            'CityName': city_reg,
            'Attraction': attraction_reg,
            'VisitMode': mode_reg,
            'AttractionType': attraction_type_reg,
            'Continent': continent_reg,
            'Region': region_reg,
            'Country': country_reg,
            'Year': year_reg,
            'Month': month_reg,
        }
        
        predicted_rating_reg = recom_dest(user_input, model, label_encoder, features, df)
        st.subheader(f"Predicted Rating: {predicted_rating_reg}")

def classification():
    st.header("Visit Mode Classification")

    # Debug info
    st.write("CityName encoder classes:", rclabel_encoder['CityName'].classes_ if isinstance(rclabel_encoder, dict) and 'CityName' in rclabel_encoder else "No CityName encoder")
    st.write("Sample CityName values:", df['CityName'].astype(str).unique()[:10])

    # Create numeric-to-text city mapping
    city_mapping = {}
    if isinstance(rclabel_encoder, dict) and 'CityName' in rclabel_encoder:
        # Convert all to strings for comparison
        encoder_cities = [str(city).lower() for city in rclabel_encoder['CityName'].classes_]
        
        for code in df['CityName'].unique():
            code_str = str(code).lower()
            # Find best matching city name
            for enc_city in rclabel_encoder['CityName'].classes_:
                enc_city_str = str(enc_city).lower()
                if enc_city_str in code_str or code_str in enc_city_str:
                    city_mapping[code] = enc_city
                    break

    st.write("City mapping (showing first 10):", dict(list(city_mapping.items())[:10]))

    # Prepare valid options
    valid_cities = sorted(df['CityName'].unique(), key=lambda x: str(x))
    valid_attractions = sorted(df['Attraction'].unique())

    # Model Evaluation Section
    with st.expander("Model Evaluation Metrics", expanded=False):
        if X_test_clf is not None and y_test_clf is not None:
            try:
                # Get VisitMode encoder
                visit_mode_encoder = None
                if isinstance(rclabel_encoder, dict):
                    visit_mode_encoder = rclabel_encoder.get('VisitMode')
                elif hasattr(rclabel_encoder, 'classes_'):
                    visit_mode_encoder = rclabel_encoder
                
                # Convert test data to proper format
                X_test_eval = X_test_clf.copy()
                y_test_eval = y_test_clf.copy()
                
                # Ensure consistent data types in features
                for col in X_test_eval.columns:
                    if X_test_eval[col].dtype == object:
                        if isinstance(rclabel_encoder, dict) and col in rclabel_encoder:
                            le = rclabel_encoder[col]
                            X_test_eval[col] = le.transform(X_test_eval[col].astype(str))
                
                # Convert y_test to numeric if needed
                if y_test_eval.dtype == object:
                    if visit_mode_encoder is not None:
                        y_test_eval = visit_mode_encoder.transform(y_test_eval.astype(str))
                    else:
                        try:
                            y_test_eval = y_test_eval.astype(int)
                        except (ValueError, TypeError):
                            y_test_eval = pd.factorize(y_test_eval)[0]
                
                # Get predictions (ensure they match y_test type)
                y_pred = rc.predict(X_test_eval)
                if visit_mode_encoder is not None and y_test_eval.dtype != object:
                    try:
                        y_pred = visit_mode_encoder.transform(visit_mode_encoder.inverse_transform(y_pred))
                    except ValueError:
                        pass
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_eval, y_pred)
                precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                
                # Display metrics
                st.subheader("Classification Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                if visit_mode_encoder is not None:
                    y_test_labels = visit_mode_encoder.inverse_transform(y_test_eval)
                    y_pred_labels = visit_mode_encoder.inverse_transform(y_pred)
                    labels = visit_mode_encoder.classes_
                else:
                    y_test_labels = y_test_eval
                    y_pred_labels = y_pred
                    labels = np.unique(np.concatenate((y_test_labels, y_pred_labels)))
                
                cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=labels, yticklabels=labels)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in model evaluation: {str(e)}")
        else:
            st.warning("Test data not available for evaluation")

    # Prediction Form
    with st.form(key='classification_form'):
        # Create dropdown showing both code and name if available
        city_options = []
        for code in valid_cities:
            display = f"{code} ({city_mapping.get(code, 'Unknown')})" if code in city_mapping else str(code)
            city_options.append(display)
        
        c_city_display = st.selectbox("City Name", options=["--Select City--"] + city_options)
        c_rating = st.slider("Rating", 1, 5, 3)
        c_attraction = st.selectbox("Attraction", options=["--Select Attraction--"] + valid_attractions)

        submit_button = st.form_submit_button("Predict Visit Mode")

    if submit_button and c_city_display != "--Select City--" and c_attraction != "--Select Attraction--":
        try:
            # Extract numeric code from display
            if '(' in c_city_display:
                c_city = c_city_display.split('(')[0].strip()
            else:
                c_city = c_city_display
            
            # Convert to original dtype if numeric
            try:
                c_city = type(df['CityName'].iloc[0])(c_city)
            except (ValueError, TypeError):
                pass

            # Get text name for encoding if mapping exists
            city_for_encoding = city_mapping.get(c_city, str(c_city))

            # Prepare input data
            input_data = {
                'CityName': city_for_encoding,
                'Rating': c_rating,
                'Attraction': c_attraction
            }

            # Encode features
            encoded = {'Rating': c_rating}
            
            if isinstance(rclabel_encoder, dict):
                if 'CityName' in rclabel_encoder:
                    try:
                        encoded['CityName'] = rclabel_encoder['CityName'].transform([city_for_encoding])[0]
                    except ValueError as e:
                        st.error(f"City encoding failed: {str(e)}. Please select a different city.")
                        return
                
                if 'Attraction' in rclabel_encoder:
                    try:
                        encoded['Attraction'] = rclabel_encoder['Attraction'].transform([c_attraction])[0]
                    except ValueError as e:
                        st.error(f"Attraction encoding failed: {str(e)}. Please select a different attraction.")
                        return

            # Make prediction
            input_df = pd.DataFrame([encoded])[RegClassfeatures]
            prediction = rc.predict(input_df)

            # Decode prediction
            if isinstance(rclabel_encoder, dict) and 'VisitMode' in rclabel_encoder:
                try:
                    prediction_label = rclabel_encoder['VisitMode'].inverse_transform(prediction)[0]
                    st.success(f"Predicted Visit Mode: {prediction_label}")
                except ValueError:
                    st.error("Model predicted an unknown visit mode category")
                    st.warning(f"Raw prediction value: {prediction[0]}")
            else:
                st.warning(f"Predicted encoded Visit Mode: {prediction[0]}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

def visualizations():
    st.header("Data Visualizations")

    # 1. Distribution of Ratings
    st.subheader("Distribution of Ratings")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Rating'], bins=5, kde=True, ax=ax1)
    st.pyplot(fig1)

    # 2. Top 10 Attraction Types
    st.subheader("Top 10 Attraction Types")
    fig2, ax2 = plt.subplots()
    top_attraction_types = type_df['AttractionType'].value_counts().head(10)
    sns.barplot(x=top_attraction_types.index, y=top_attraction_types.values, ax=ax2)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

    # 3. Top 10 Visit Modes
    st.subheader("Top 10 Visit Modes")
    fig3, ax3 = plt.subplots()
    top_visit_modes = mode_df['VisitMode'].value_counts().head(10)
    sns.barplot(x=top_visit_modes.index, y=top_visit_modes.values, ax=ax3)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)

    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

    # 5. Top 10 cities
    st.subheader("Top 10 Cities")
    fig5, ax5 = plt.subplots()
    top_cities = df['CityName'].value_counts().head(10)
    sns.barplot(x=top_cities.index, y=top_cities.values, ax=ax5)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)

def about():
    st.header("About the App")
    st.write("This app provides tourism recommendations and predictions using machine learning models.")
    st.write("It includes features for recommending destinations based on user preferences, predicting ratings, and classifying visit modes.")
    st.write("The models are trained on a tourism dataset, and the app uses Streamlit for its user interface.")

def help():
    st.header("Help")
    st.write("For recommendations, enter the User ID, City Name, Attraction Name, select Visit Mode, and Attraction Type, then click 'Recommend'.")
    st.write("For classification, enter the User ID, City Name, Rating, Attraction Name, and select Attraction Type, then click 'Predict Visit Mode'.")
    st.write("The 'Visualizations' section displays various graphs generated from the dataset.")

def main():
    option = st.sidebar.radio("Select an Option", ["Recommendation", "Regression", "Classification", "Visualizations", "About", "Help"], index=0)

    if option == "Recommendation":
        recommendation()
    elif option == "Regression":
        regression()
    elif option == "Classification":
        classification()
    elif option == "Visualizations":
        visualizations()
    elif option == "About":
        about()
    elif option == "Help":
        help()
  
if __name__ == "__main__":
    main()