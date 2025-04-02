# Kaggle Student Performance Prediction

This project is based on the **Kaggle Student Performance** dataset, which is used to predict students' final grades based on various features like study time, past grades, and school-related factors. The project includes several machine learning models to predict student performance and compares them after hyperparameter tuning. The app is deployed using Streamlit for interactive visualization.

## [Live Demo](https://kaggle-student-performance-varunrao.streamlit.app/)
## [Link to kaggle Notebook](https://www.kaggle.com/code/varunraosfanlkan/notebook3ac0f15a42)
You can interact with the live model and test various inputs on the deployed Streamlit app.

## Project Overview

The goal of this project is to build and compare multiple regression models that can predict student performance based on different features.

### Models Used:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Ridge Regression**

The models were evaluated based on their performance on the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

## Final Model Comparison

### 1. **Linear Regression**

- **MAE**: 1.612
- **MSE**: 4.087
- **R²**: 0.989

**Strengths**: Linear Regression performed exceptionally well with a near-perfect R² score, indicating that it explains about 98.9% of the variance in the target variable. It is simple and interpretable.

**Conclusion**: Linear Regression is a solid choice for this dataset, especially for its simplicity and ease of interpretation.

---

### 2. **Random Forest (Best Model)**

- **MAE**: 1.721
- **MSE**: 4.672
- **R²**: 0.987

**Strengths**: Random Forest is a non-linear model that captures more complex interactions between the features. It achieved a very high R² score (98.7%) but performed slightly worse than Linear Regression.

**Conclusion**: Random Forest is a powerful model for capturing non-linearities but doesn't perform significantly better in this specific case. It might perform better with different datasets or hyperparameters.

---

### 3. **Ridge Regression (Best Model)**

- **MAE**: 1.612
- **MSE**: 4.089
- **R²**: 0.989

**Strengths**: Ridge Regression is similar to Linear Regression but with regularization to prevent overfitting. The performance is almost identical to Linear Regression, making it a strong contender.

**Conclusion**: Ridge Regression behaves similarly to Linear Regression but with the added benefit of regularization, which helps when dealing with correlated features.

---

## Key Takeaways

- **Linear Regression** and **Ridge Regression** performed nearly identically, with Ridge providing a slight advantage in preventing overfitting due to regularization.
- **Random Forest** showed slightly worse performance than Linear Regression and Ridge Regression in this case, but it might be more useful for more complex or non-linear relationships in different datasets.
- All models performed very well with high **R² values**, indicating that they fit the data well and are capable of making accurate predictions.

## Final Recommendation

- If **simplicity** and **interpretability** are your priorities, **Linear Regression** or **Ridge Regression** are the best choices, as they provide nearly identical results and are easy to interpret.
- **Random Forest** could be a good option if you anticipate more complex relationships in future datasets or if you want to explore non-linear interactions, though it didn’t outperform the linear models in this specific instance.

## Installation & Setup

To run this project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/kaggle-student-performance.git
```

### 2. Navigate into the project directory
```bash
cd kaggle-student-performance
```
### 3. Install the dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Streamlit app
```bash
streamlit run app.py
```
## Libraries Used
1. **Streamlit**: For building the interactive web app.

2. **Scikit-learn**: For building and evaluating machine learning models.

3. **Pandas**: For data manipulation and cleaning.

4. **NumPy**: For numerical operations.

5. **Matplotlib / Plotly**: For data visualization.
## Final Thoughts
This project demonstrates the power of both simple and complex regression models to solve a real-world problem. While Linear Regression and Ridge Regression provide excellent performance and are easy to interpret, Random Forest provides flexibility for more complex datasets. This work highlights how machine learning can be applied to predict outcomes, even when starting from simple data.
