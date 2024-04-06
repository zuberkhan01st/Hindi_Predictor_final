import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from googletrans import Translator

def main():
    
    st.title('Health Genie - बीमारी की भविष्यवाणी')

    # Load the dataset for symptoms
    symptoms_data = pd.read_csv('Testing.csv')

    # Select first 15 symptoms
    selected_features = symptoms_data.columns[1:16]

    # Separate features (X) and target (y)
    X = symptoms_data[selected_features]
    y = symptoms_data['prognosis']

    # Load the dataset for diseases and descriptions
    diseases_data = pd.read_csv('Training.csv')

    # Create a dictionary mapping diseases to their descriptions
    disease_info = dict(zip(diseases_data['Disease'], diseases_data['Description']))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Map symptom severity levels to numerical values
    severity_mapping = {'नहीं': 0, 'हाँ': 1}

    # Function for translating text to Hindi using Google Translate
    def translate_to_hindi(text):
        translator = Translator()
        translation = translator.translate(text, dest='hi')
        return translation.text

    # Translate sidebar content to Hindi
    sidebar_content = """
    ## विकल्प
    भाषा को हिंदी में अनुवाद करें
    """
    
    st.sidebar.header('उन लक्षणों का चयन करें जो आप अनुभव कर रहे हैं')
    # Get user input
    user_input = {}
    for feature in selected_features:
        severity_level = st.sidebar.selectbox(f'{translate_to_hindi(feature)} के लिए गंभीरता का चयन करें:', ['नहीं', 'हाँ'])
        user_input[feature] = severity_mapping[severity_level]

    # Convert user input to DataFrame
    user_data = pd.DataFrame(user_input, index=[0])

    # Predict disease
    if st.button('भविष्यवाणी करें'):
        disease_prediction = clf.predict(user_data)
        predicted_disease = disease_prediction[0]
        
        st.write(f'पूर्वानुमानित बीमारी: {predicted_disease}')

        # Display additional information about the predicted disease
        if predicted_disease in disease_info:
            st.write('अतिरिक्त जानकारी:')
            st.write(disease_info.get(predicted_disease, 'अतिरिक्त जानकारी उपलब्ध नहीं है।'))
        else:
            st.write('अतिरिक्त जानकारी उपलब्ध नहीं है।')

        
        st.markdown("""
        *अस्वीकृति:* यह बीमारी की भविष्यवाणी के लिए एक मशीन शिक्षा आधारित मॉडल है। यह अनुमान प्रदान कर सकता है, लेकिन हमेशा सटीक नहीं होता है। सटीक चिकित्सा सलाह के लिए, कृपया किसी योग्य स्वास्थ्य पेशेवर से संपर्क करें।
        """)



if __name__ == "__main__":
    main()