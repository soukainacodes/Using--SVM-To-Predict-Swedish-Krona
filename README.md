# Predicting Exchange Rates Using SVM and Different Machine Learning Methods

## Aalto University, August 2022

### Introduction
Currency fluctuations are a natural outcome of floating exchange rates, which is the norm for most major economies. Various factors influence exchange rates, including a country's economic performance, inflation outlook, interest rate differentials, and capital flows.

The main goal of this project is to help brokers, businesses, and governments make better financial decisions, control cash flow, and prevent financial crimes like money laundering. By leveraging machine learning techniques, we aim to predict the future exchange rates of the Swedish Krona (SEK) based on historical data from the European Central Bank (ECB).

We will use exchange rates from a specific day as features, with the SEK-EUR exchange rate as our main label. Two machine learning models will be applied, and their performance compared using loss functions to determine the most efficient model.

### Problem Formulation
The problem consists of three key components:
1. **Data Points**: Exchange rates from EUR to DKK, EUR to USD, and EUR to NOK serve as features, while EUR to SEK is the target label. The dataset includes records from January 4, 1999, to August 8, 2022, totaling 6,048 data points.
2. **Models**: Machine learning models are used to fit the data and make predictions.
3. **Loss Functions**: These are used to evaluate model performance by measuring prediction errors.

The following loss functions are utilized:
- **Mean Squared Error (MSE)**: Sensitive to outliers due to squared penalties but ensures a single global minimum.
- **Huber Loss**: A combination of MSE and Mean Absolute Error (MAE) that reduces sensitivity to outliers.

### Methods
We used two machine learning models for predicting the SEK exchange rate:
1. **Multiple Linear Regression**: 
   - Uses linear mapping to predict the target variable.
   - Implemented using `sklearn.linear_model.LinearRegression`.
   - Loss function: Mean Squared Error (MSE).
2. **Huber Regression**:
   - Also a linear model but with a loss function more robust to outliers.
   - Implemented using `sklearn.linear_model.HuberRegressor`.
   - Loss function: Huber Loss.

**Data Source:** European Central Bank (https://sdw.ecb.europa.eu/)
- Training Set: 1999-01 to 2019-12
- Validation Set: 2020-01 to 2021-12
- Test Set: 2022-01 to 2022-08

Polynomial models were not used due to the risk of overfitting.

### Results
#### Training and Validation Errors:
| Method  | Training Error | Validation Error |
|---------|---------------|-----------------|
| Linear  | 0.1692        | 0.1473          |
| Huber   | 0.0874        | 0.0228          |

Huber Regression shows lower errors, suggesting better robustness against outliers.

#### Residual Analysis:
- Both models show residuals skewed to the right, indicating some limitations.
- Despite better validation performance, Huber Regression has a higher test error (0.0951) than Linear Regression (0.07).

### Conclusion
- **Huber Regression** performs better during training and validation but has a higher test error.
- **Linear Regression** is recommended for real-world applications due to better generalization on unseen data.
- Future research should explore alternative models and time series methods for improved forecasting.

### References
- Alexander Jung, "Three Components of Machine Learning", Aalto University, 2022.
- [Loss Functions Overview](https://www.numpyninja.com/amp/loss-functions-when-to-use-which-one)
- [European Central Bank Data](https://sdw.ecb.europa.eu/)

### Appendix
The full implementation, dataset, and results can be found in the included notebook.
