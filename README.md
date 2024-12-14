Diabetes Prediction using Machine Learning

This project leverages Machine Learning techniques to predict the likelihood of diabetes in individuals based on medical data. Using the Pima Diabetes Dataset from Kaggle, the project implements a Support Vector Machine (SVM) classifier to achieve high prediction accuracy.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Features

Dataset: Pima Diabetes Dataset from Kaggle.

Data Preprocessing: Standardization and feature scaling using StandardScaler.

Model Training: Support Vector Machine (SVM) classifier for predictive analysis.

Performance Evaluation: Calculation of accuracy on training and test datasets.

User Input Prediction: Real-time prediction of diabetes based on user-provided medical information.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Technologies Used

The project uses the following Python libraries:

NumPy: For numerical computations.

Pandas: For data manipulation and analysis.

Scikit-learn: For preprocessing, model training, and evaluation.

Warnings: To suppress unnecessary warnings.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dataset

The dataset used in this project is the Pima Diabetes Dataset, sourced from Kaggle. It contains several medical predictor variables and one target variable:

Predictor Variables: Includes features like glucose level, blood pressure, BMI, and age.

Target Variable: Indicates whether the individual is diabetic (1) or not (0).

Downloading the Dataset

To download the dataset, use the Kaggle API:

Install the Kaggle Python package:

pip install kaggle

Authenticate using your Kaggle API key. Save your kaggle.json file in the .kaggle directory in your home folder.

Extract the dataset:

import zipfile

data_path = "/content/diabetes.csv"
with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall("dataset-directory")
    print("Dataset extracted successfully.")

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model Workflow

Data Preprocessing:

Standardize the dataset using StandardScaler to normalize feature values.

Split the dataset into training and testing sets using train_test_split.

Model Training:

Train an SVM classifier on the training data.

Model Evaluation:

Calculate accuracy scores for training and test data.

Prediction:

Accept user input, preprocess it, and predict diabetes status.

Key Code Snippets

Training the Model

classifier.fit(X_train, Y_train)

Accuracy Calculation

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

User Input Prediction

input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Results

Training Accuracy: Achieved high accuracy during training.

Test Accuracy: Robust performance on unseen test data.

Prediction: Effective prediction for user-provided inputs.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How to Run

Clone the repository:

git clone https://github.com/Diabetes-prediction-using-SVM.git

Install the required libraries:

pip install -r requirements.txt

Run the notebook:

jupyter notebook diabetes_prediction.ipynb

Provide input data for prediction and see the results.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Future Improvements

Test on larger datasets for better generalization.

Integrate advanced models like Random Forest or XGBoost.

Build a web application for real-time predictions.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

License

This project is open-source and available under the MIT License.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Acknowledgments

Kaggle: For providing the dataset.

Scikit-learn Documentation: For model implementation guidance.
