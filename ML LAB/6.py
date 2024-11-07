import bayespy as bp
import numpy as np
import csv
from colorama import init
from colorama import Fore, Back, Style

init()

# Define Parameter Enum values
ageEnum = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
genderEnum = {'Male': 0, 'Female': 1}
familyHistoryEnum = {'Yes': 0, 'No': 1}
dietEnum = {'High': 0, 'Medium': 1, 'Low': 2}
lifeStyleEnum = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedetary': 3}
cholesterolEnum = {'High': 0, 'BorderLine': 1, 'Normal': 2}
heartDiseaseEnum = {'Yes': 0, 'No': 1}

# Load the data from the CSV file with tab delimiter
data = []
with open('heart_disease_data.csv') as csvfile:
    lines = csv.reader(csvfile, delimiter='\t')
    next(lines)  # Skip the header row
    for x in lines:
        # Map each row to the respective enum values
        data.append([
            ageEnum[x[0]], 
            genderEnum[x[1]], 
            familyHistoryEnum[x[2]], 
            dietEnum[x[3]], 
            lifeStyleEnum[x[4]], 
            cholesterolEnum[x[5]], 
            heartDiseaseEnum[x[6]]
        ])

# Convert data to a numpy array
data = np.array(data)
N = len(data)

# Initialize Bayesian nodes
p_age = bp.nodes.Dirichlet(1.0 * np.ones(5))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:, 0])

p_gender = bp.nodes.Dirichlet(1.0 * np.ones(2))
gender = bp.nodes.Categorical(p_gender, plates=(N,))
gender.observe(data[:, 1])

p_familyhistory = bp.nodes.Dirichlet(1.0 * np.ones(2))
familyhistory = bp.nodes.Categorical(p_familyhistory, plates=(N,))
familyhistory.observe(data[:, 2])

p_diet = bp.nodes.Dirichlet(1.0 * np.ones(3))
diet = bp.nodes.Categorical(p_diet, plates=(N,))
diet.observe(data[:, 3])

p_lifestyle = bp.nodes.Dirichlet(1.0 * np.ones(4))
lifestyle = bp.nodes.Categorical(p_lifestyle, plates=(N,))
lifestyle.observe(data[:, 4])

p_cholesterol = bp.nodes.Dirichlet(1.0 * np.ones(3))
cholesterol = bp.nodes.Categorical(p_cholesterol, plates=(N,))
cholesterol.observe(data[:, 5])

# Prepare nodes and establish edges
p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = bp.nodes.MultiMixture([age, gender, familyhistory, diet, lifestyle, cholesterol], bp.nodes.Categorical, p_heartdisease)
heartdisease.observe(data[:, 6])
p_heartdisease.update()

# Interactive test loop
m = 0
while m == 0:
    print("\nEnter the values for testing Heart Disease probability.")
    age_input = int(input('Enter Age: ' + str(ageEnum) + ' '))
    gender_input = int(input('Enter Gender: ' + str(genderEnum) + ' '))
    family_history_input = int(input('Enter FamilyHistory: ' + str(familyHistoryEnum) + ' '))
    diet_input = int(input('Enter Diet: ' + str(dietEnum) + ' '))
    lifestyle_input = int(input('Enter LifeStyle: ' + str(lifeStyleEnum) + ' '))
    cholesterol_input = int(input('Enter Cholesterol: ' + str(cholesterolEnum) + ' '))
    
    # Calculate the probability of heart disease based on input values
    res = bp.nodes.MultiMixture(
        [age_input, gender_input, family_history_input, diet_input, lifestyle_input, cholesterol_input],
        bp.nodes.Categorical, p_heartdisease
    ).get_moments()[0][heartDiseaseEnum['Yes']]
    
    print("Probability(HeartDisease) = " + str(res))
    
    # Option to continue or exit
    m = int(input("Enter 0 to continue or 1 to exit: "))
