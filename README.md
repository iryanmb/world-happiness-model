# World Happiness Prediction Model

A machine learning model that predicts national happiness levels using socioeconomic indicators from the 2018 World Happiness Report. Achieved **89.4% accuracy (R² = 0.894)** with **RMSE of 0.3578**.

## Problem Statement
Understanding what drives national happiness enables governments and organizations to make data-driven policy decisions and optimize resource allocation for maximum societal well-being impact.

## Dataset & Approach
- **Data:** 1,562 records across 136 countries from WHR 2018  
- **Target:** Life Ladder (happiness index score)  
- **Method:** Tree-based regression models with strategic feature engineering  

## Key Technical Decisions

### Data Preprocessing
- Removed 6 features with low correlation (< 0.2) or excessive missing values (> 60%)  
- Applied median imputation for right-skewed distributions (skewness = 0.876)  
- Retained outliers in corruption perception as valuable country-specific data  
- Selected tree-based models for robust outlier handling  

### Model Selection
Compared three algorithms on a 60/20/20 train/validation/test split:

| Model            | RMSE    | R² Score | Notes                           |
|------------------|---------|----------|---------------------------------|
| Decision Tree    | 0.4890  | 0.8158   | Baseline                        |
| Random Forest    | 0.3578  | 0.8940   | **Selected**                    |
| Gradient Boosting| 0.3721  | 0.8880   | Slight overfitting              |

**Why Random Forest:** Best validation performance with consistent test results, superior outlier handling through ensemble approach.

## Results & Insights

### Primary Happiness Predictors (Feature Correlation)
1. **Log GDP per capita** (0.796) — Economic prosperity  
2. **Social support** (0.760) — Community networks  
3. **Life expectancy** (0.725) — Health infrastructure  
4. **Freedom of choice** (0.541) — Personal autonomy  
5. **Low corruption** (-0.425) — Institutional trust  

### Model Performance
- **Final Test Accuracy:** 89.4% of happiness variance explained  
- **Prediction Error:** 0.36 points on a 10-point happiness scale  
- **Validation:** 10-fold cross-validation confirmed generalizability  

## Business Applications
- **Government Policy:** Predict happiness impact of economic and social interventions  
- **Development Organizations:** Optimize aid allocation based on happiness ROI  
- **Corporate Social Responsibility:** Guide community investment strategies  
