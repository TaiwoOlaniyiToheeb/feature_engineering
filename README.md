# NBA Player Feature Engineering for Career Longevity Prediction

## Project Overview

This project performs feature engineering and preprocessing on an NBA player dataset to prepare it for machine learning modeling. The objective is to predict whether a player will remain active in the NBA for at least five years using player performance statistics.

The project focuses on cleaning the data, reducing multicollinearity, creating new features, and preparing the dataset for predictive modeling.

---

## Dataset

The dataset contains historical NBA player statistics, including scoring, rebounding, shooting efficiency, and defensive performance.

### Target Variable

* **target_5yrs**

  * `1`: Player remained in the NBA for at least five years.
  * `0`: Player did not remain in the NBA for at least five years.

This column serves as the dependent variable for machine learning models.

---

## Project Objectives

The main objectives of this project were to:

* Isolate and define the target variable (`target_5yrs`).
* Remove non-predictive columns.
* Identify and address multicollinearity.
* Engineer meaningful features.
* Handle missing values.
* Prepare a clean dataset for machine learning.

---

## Data Cleaning

### Removing Non-Predictive Features

The following columns were removed because they do not contribute to prediction and may introduce noise or data leakage:

* `Unnamed: 0`
* `name`

```python
df.drop(['Unnamed: 0', 'name'], axis=1, inplace=True)
```

---

## Correlation Analysis and Multicollinearity

A correlation matrix and heatmap were generated to identify highly correlated features.

Features with correlation coefficients greater than 0.90 were considered redundant.

The following variables were removed:

* `fgm`
* `fga`
* `fta`
* `3pa`
* `oreb`
* `dreb`

These variables were highly correlated with more informative or aggregated features such as:

* `pts`
* `reb`
* `ftm`
* `3p_made`

Reducing multicollinearity improves model stability and interpretability.

```python
df.drop(['fgm', 'fga', 'fta', '3pa', 'dreb', 'oreb'], axis=1, inplace=True)
```

---

## Feature Engineering

Two new features were created to better capture player efficiency and overall contribution.

### 1. Points Per Minute (PPM)

Measures scoring efficiency relative to playing time.

```python
df['points_per_min'] = df['pts'] / df['min']
```

### 2. Efficiency Rating

Provides a simplified measure of overall player performance.

```python
df['efficiency_rating'] = (
    df['pts']
    + df['reb']
    + df['ast']
    + df['stl']
    + df['blk']
    - df['tov']
)
```

These engineered features provide richer information than raw statistics alone.

---

## Missing Value Treatment

Missing values in performance columns were examined using:

```python
df.isnull().sum()
```

Any missing values were handled using median imputation to maintain data integrity while minimizing the influence of outliers.

```python
df.fillna(df.median(numeric_only=True), inplace=True)
```

If no missing values were found, no further imputation was necessary.

---

## Final Dataset

The final dataset contains:

* Reduced multicollinearity
* Engineered performance metrics
* Cleaned and machine-learning-ready features

This dataset is suitable for classification models aimed at predicting NBA career longevity.


---

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Author

Taiwo Olaniyi Toheeb
Data Scientist and Transportation Analyst
