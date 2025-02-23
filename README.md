# Spaceship-Titanic-ML-Model
# Code Description

This repository contains the code used for the **Spaceship Titanic** competition on Kaggle. The goal of the competition is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. Below is a detailed description of the code and its workflow.

---

## **Code Overview**

The code is written in Python using Jupyter Notebook and leverages popular data science libraries such as `pandas`, `numpy`, and `lightgbm`. The workflow includes data preprocessing, feature engineering, model training, and prediction generation.

---

## **Steps in the Code**

### 1. **Importing Libraries**
   - The necessary libraries are imported:
     ```python
     import pandas as pd
     import numpy as np
     from sklearn.preprocessing import LabelEncoder
     from lightgbm import LGBMClassifier
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import accuracy_score
     ```

### 2. **Loading Data**
   - The training and test datasets are loaded into DataFrames:
     ```python
     df1 = pd.read_csv("C:\\Users\\Acer\\Desktop\\spaceship-titanic\\train.csv")
     df2 = pd.read_csv("C:\\Users\\Acer\\Desktop\\spaceship-titanic\\test.csv")
     ```

### 3. **Data Preprocessing**
   - The `Transported` column is added to the test dataset with a default value of `False` to facilitate concatenation:
     ```python
     if 'Transported' not in df2.columns:
         df2['Transported'] = False
     ```
   - The training and test datasets are concatenated for consistent preprocessing:
     ```python
     df = pd.concat([df1, df2], axis=0)
     ```

### 4. **Feature Engineering**
   - The `Cabin` column is split into three separate columns (`Deck`, `Num`, and `Side`) for better feature representation:
     ```python
     df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
     df = df.drop(columns='Cabin')
     ```
   - Missing values in numerical columns are filled with the mean of the respective column:
     ```python
     float_col = ['Age', 'RoomService', 'FoodCourt', 'Spa', 'VRDeck', 'ShoppingMall']
     for i in float_col:
         df[i] = df[i].fillna(np.round(df[i].mean()))
     ```
   - Missing values in categorical columns are filled with the mode of the respective column:
     ```python
     for i in df.select_dtypes(include='object'):
         df[i].fillna(df[i].mode()[0], inplace=True)
     ```

### 5. **Label Encoding**
   - Categorical columns are encoded using `LabelEncoder`:
     ```python
     label = LabelEncoder()
     col_to_label = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
     for i in col_to_label:
         df[i] = label.fit_transform(df[i])
     ```

### 6. **Dropping Unnecessary Columns**
   - Columns like `PassengerId` and `Name` are dropped as they are not useful for modeling:
     ```python
     df.drop(columns='PassengerId', inplace=True)
     df.drop(columns='Name', inplace=True)
     ```

### 7. **Splitting Data**
   - The concatenated dataset is split back into training and test sets:
     ```python
     train, test = df[:df1.shape[0]], df[df1.shape[0]:]
     test = test.drop(columns='Transported')
     ```

### 8. **Model Training**
   - A LightGBM classifier is used for training:
     ```python
     X = train.drop(columns='Transported')
     y = train['Transported']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     model = LGBMClassifier()
     model.fit(X_train, y_train)
     ```
   - The model's accuracy is evaluated on the test set:
     ```python
     pred = model.predict(X_test)
     accuracy_score(y_test, pred)
     ```

### 9. **Generating Predictions**
   - Predictions are generated for the test dataset and saved to a CSV file:
     ```python
     y_pred = model.predict(test)
     final = pd.DataFrame()
     df_dummy = pd.read_csv("test.csv")
     final['PassengerId'] = df_dummy['PassengerId']
     final['Transported'] = y_pred
     final.to_csv('Sample1.csv', index=False)
     ```

---

## **Key Features**
- **Feature Engineering**: The `Cabin` column is split into `Deck`, `Num`, and `Side` for better feature representation.
- **Handling Missing Values**: Missing values are filled using mean (for numerical columns) and mode (for categorical columns).
- **LightGBM Model**: A LightGBM classifier is used for training, achieving an accuracy of approximately **81%** on the validation set.

---

## **How to Use**
1. Clone the repository.
2. Ensure the dataset files (`train.csv` and `test.csv`) are placed in the correct directory.
3. Run the Jupyter Notebook to preprocess the data, train the model, and generate predictions.
4. The predictions will be saved in `Sample1.csv`.

---

## **Dependencies**
- Python 3.x
- Libraries: `pandas`, `numpy`, `lightgbm`, `scikit-learn`

---

## **Future Improvements**
- Hyperparameter tuning for the LightGBM model.
- Experimenting with other models like XGBoost or CatBoost.
- Additional feature engineering to improve model performance.

---

This code provides a solid baseline for the Spaceship Titanic competition and can be further improved for better performance.
