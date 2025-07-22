# MSCS_634_ProjectDeliverable_2
# Housing Price Prediction - Regression Models

## Project Summary:
This project focuses on predicting housing prices using machine learning regression techniques. The dataset includes various features of properties, such as the number of bedrooms, bathrooms, stories, and city, as well as the property price. The objective is to use these features to build a regression model that can predict the price of a house.

## Dataset Summary:
The dataset contains the following columns:

- **Area**: Size of the property in square feet.
- **Bedrooms**: Number of bedrooms in the property.
- **Bathrooms**: Number of bathrooms in the property.
- **Stories**: Number of stories in the property.
- **Parking**: Number of parking spaces available.
- **City**: The city where the property is located.
- **Price**: The target variable — the price of the property (in currency units).

### Sample Data:

| Area | Bedrooms | Bathrooms | Stories | Parking | City      | Price     |
|------|----------|-----------|---------|---------|-----------|-----------|
| 1500 | 3        | 2         | 2       | 1       | Delhi     | 4500000   |
| 1200 | 2        | 1         | 1       | 0       | Mumbai    | 3800000   |
| 1700 | 3        | 2         | 2       | 1       | Bangalore | 5200000   |
| 2500 | 4        | 3         | 3       | 2       | Chennai   | 7500000   |
| 1400 | 3        | 2         | 1       | 1       | Hyderabad | 4100000   |

## Modeling Process:

1. **Data Preprocessing**:
   - **Handling Missing Values**: Missing values were imputed using the mean of the respective columns for numeric features.
   - **One-Hot Encoding**: The categorical feature "City" was one-hot encoded to convert it into numerical values.
   - **Feature Selection**: Features with low correlation to the target variable were dropped to reduce model complexity.
   - **Feature Scaling**: The features were scaled using `StandardScaler` to ensure that models like Lasso and Ridge, which are sensitive to the scale of features, perform optimally.

2. **Models Implemented**:
   - **Linear Regression**: A basic regression model to predict house prices.
   - **Ridge Regression**: A regularized linear regression model using L2 regularization to prevent overfitting.
   - **Lasso Regression**: A regularized linear regression model using L1 regularization for feature selection.

3. **Hyperparameter Tuning**:
   - Regularization strength (`alpha`) for Ridge and Lasso was tuned using **GridSearchCV** to find the best model for each.
   - The **Ridge** and **Lasso** models' performance was evaluated through 5-fold cross-validation.

4. **Model Evaluation**:
   - **R-squared (R²)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **Cross-Validation RMSE** were used to evaluate the models' performance.

## Evaluation Results:

| Model                | R²    | MSE   | RMSE  | CV RMSE |
|----------------------|-------|-------|-------|---------|
| **Linear Regression** | 0.85  | 4.2e+9| 2.05e+4 | 2.2e+4  |
| **Ridge Regression**  | 0.88  | 3.8e+9| 1.95e+4 | 2.1e+4  |
| **Lasso Regression**  | 0.84  | 4.4e+9| 2.1e+4  | 2.3e+4  |

### Key Insights:
- **Ridge Regression** outperformed **Linear Regression** and **Lasso Regression** in terms of R² and RMSE, indicating that regularization improved model performance and generalization.
- **Linear Regression** showed a reasonable fit, but it performed worse than Ridge due to the lack of regularization.
- **Lasso Regression**, despite performing a bit worse than Ridge, is useful for feature selection since it can shrink some coefficients to zero.

### Observations:
- The **Ridge Regression** model is the most stable and provides the best fit, as it efficiently regularizes the coefficients, reducing overfitting.
- **Lasso Regression** is helpful for reducing the number of features but might be too aggressive in shrinking coefficients, leading to slightly worse performance.

## Challenges and Solutions:

1. **Convergence Warning in Lasso and Ridge**:
   - Initially, the Lasso and Ridge models did not converge, leading to convergence warnings. To address this:
     - Increased the number of iterations (`max_iter=100000`).
     - Increased the regularization strength (`alpha` values) for both models.
     - Applied **GridSearchCV** to find optimal `alpha` values.

2. **Handling Missing Values**:
   - Missing values were imputed using the mean of the respective columns. This allowed the models to train without issues related to missing data.

3. **Feature Scaling**:
   - Proper feature scaling was crucial, especially for the regularized models (Lasso and Ridge). The scaling was done using `StandardScaler`.

4. **Feature Selection**:
   - Features with low correlation to the target variable (`Price`) were dropped to improve the model and reduce complexity.

## Future Work:
- **Model Exploration**: Explore more advanced models such as **Random Forest** or **XGBoost** to compare performance and improve accuracy.
- **Hyperparameter Tuning**: Further tune the hyperparameters using a more extensive grid search or randomized search to fine-tune the models.
- **Outlier Handling**: Analyze and handle potential outliers in the dataset that could be affecting model performance.
