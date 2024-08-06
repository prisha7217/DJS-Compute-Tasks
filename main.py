import joblib
import streamlit as st
import pandas as pd

try:
    model = joblib.load('xgb_model.pkl')
    st.write("Model Loaded Successfully")
except Exception as e:
    st.write(f"Error Loading Model: {e}")
    st.stop()

income_mapping = {
    1: '>50K',
    0: '<=50K'
}
    
def main():
    st.title("Income Predictor")

    #age, workclass, education, marital status, relationship, capital gain, capital loss, hours per week, native country, occupation, race, sex
    age = st.number_input('Age', min_value=0, max_value=100)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=1000000)
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=1000000)
    hours = st.number_input('Hours Per Week', min_value=1, max_value=100)

    workclass = st.selectbox('Workclass', ['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc',
       'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox('Education', ['HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate',
       'Prof-school', 'Bachelors', 'Masters', '11th', 'Assoc-acdm',
       'Assoc-voc', '1st-4th', '5th-6th', '12th', '9th', 'Preschool'])
    marital = st.selectbox('Marital Status', ['Widowed', 'Divorced', 'Separated', 'Never-married',
       'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'])
    relationship = st.selectbox('Relationship', ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative',
       'Husband', 'Wife'])
    countries = ['United-States', 'Mexico', 'Greece', 'Vietnam', 'China',
       'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada',
       'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran',
       'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba',
       'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
       'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
       'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland',
       'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)']
    country = st.selectbox('Native Country', countries)

    occupations = ['Adm-clerical','Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
    occupation = st.selectbox('Occupation', occupations)
    races = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
    race = st.selectbox('Race', races)
    sexes = ['Female', 'Male']
    sex = st.selectbox('Gender', sexes)

    one_hot_encoded_occupation = [0] * len(occupations)
    occupation_index = occupation.index(occupation)
    one_hot_encoded_occupation[occupation_index] = 1

    one_hot_encoded_race = [0] * len(races)
    race_index = races.index(race)
    one_hot_encoded_race[race_index] = 1

    one_hot_encoded_gender = [0] * len(sexes)
    gender_index = sexes.index(sex)
    one_hot_encoded_gender[gender_index] = 1

    data = pd.DataFrame({
        'age': [age],
        'workclass': [{'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp-inc': 3, 'Self-emp-not-inc': 4, 'State-gov': 5, 'Without-pay': 6}[workclass]],
        'education': [{'HS-grad': 9, 'Some-college': 10, '7th-8th': 4, '10th': 6, 'Doctorate': 16, 'Prof-school': 15, 'Bachelors': 13, 'Masters': 14,  '11th': 7,'Assoc-acdm': 12, 'Assoc-voc': 11, '1st-4th': 2, '5th-6th': 3, '12th': 8, '9th': 5, 'Preschool': 1}[education]],
        'marital.status': [{'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}[marital]],
        'relationship': [{'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}[relationship]],
        'capital.gain': [capital_gain],
        'capital.loss': [capital_loss], 
        'hours.per.week': [hours],
        'native.country': [{'Cambodia': 1, 'Canada': 2, 'China': 3, 'Columbia': 4, 'Cuba': 5, 'Dominican-Republic': 6, 'Ecuador': 7, 'El-Salvador': 8, 'England': 9, 'France': 10, 'Germany': 11, 'Greece': 12, 'Guatemala': 13, 'Haiti': 14, 'Holand-Netherlands': 15, 'Honduras': 16, 'Hong': 17, 'Hungary': 18, 'India': 19, 'Iran': 20, 'Ireland': 21, 'Italy': 22, 'Jamaica': 23, 'Japan': 24, 'Laos': 25, 'Mexico': 26, 'Nicaragua': 27, 'Outlying-US(Guam-USVI-etc)': 28, 'Peru': 29, 'Philippines': 30, 'Poland': 31, 'Portugal': 32, 'Puerto-Rico': 33, 'Scotland': 34, 'South': 35, 'Taiwan': 36, 'Thailand': 37, 'Trinadad&Tobago': 38, 'United-States': 39, 'Vietnam': 40, 'Yugoslavia': 41}[country]],
        'occupation_Adm-clerical':  [one_hot_encoded_occupation[0]],
        'occupation_Armed-Forces': [one_hot_encoded_occupation[1]],
        'occupation_Craft-repair': [one_hot_encoded_occupation[2]],
        'occupation_Exec-managerial': [one_hot_encoded_occupation[3]],
        'occupation_Farming-fishing': [one_hot_encoded_occupation[4]],
        'occupation_Handlers-cleaners': [one_hot_encoded_occupation[5]],
        'occupation_Machine-op-inspct': [one_hot_encoded_occupation[6]],
        'occupation_Other-service': [one_hot_encoded_occupation[7]],
        'occupation_Priv-house-serv': [one_hot_encoded_occupation[8]],
        'occupation_Prof-specialty': [one_hot_encoded_occupation[9]],
        'occupation_Protective-serv': [one_hot_encoded_occupation[10]],
        'occupation_Sales': [one_hot_encoded_occupation[11]],
        'occupation_Tech-support': [one_hot_encoded_occupation[12]],
        'occupation_Transport-moving': [one_hot_encoded_occupation[13]],
        'race_Amer-Indian-Eskimo': [one_hot_encoded_race[0]],
        'race_Asian-Pac-Islander': [one_hot_encoded_race[1]],
        'race_Black': [one_hot_encoded_race[2]],
        'race_Other': [one_hot_encoded_race[3]],
        'race_White': [one_hot_encoded_race[4]],
        'sex_Female': [one_hot_encoded_gender[0]],
        'sex_Male': [one_hot_encoded_gender[1]]
    })

    st.write('Input Data for Prediction')
    st.write(data)

    try:
        prediction = model.predict(data)
        income = income_mapping.get(prediction[0], 'Uknown Income')
        st.write(f"Predicted Income: {income}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()

    