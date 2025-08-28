# **Video Game Sales Forecasting**

📌 Project Overview

This project analyzes global video game sales data to identify the strongest predictors of commercial success and to forecast sales trends over time. Using multiple classification and time series frameworks in R, the study provides actionable recommendations for developers and publishers aiming to maximize profitability in an evolving gaming market
.

🔎 Dataset

Source: Historical sales dataset covering 11,470 video games (1980s–2010s).

Features: Rank, Name, Platform, Year, Genre, Publisher, regional sales (NA, EU, JP, Other), and Global Sales.

Business Definition of Success: Games selling >0.5 million global units.

🛠️ Methodology
Exploratory Data Analysis (EDA)

Average global sales: 0.60M units; median: 0.17M units → highly skewed distribution.

Blockbuster Effect: A handful of games dominate overall sales.

Genre: Action, Sports, Shooter → highest potential.

Platform: Legacy platforms (Game Boy, NES) outsold modern consoles historically.

Publisher: Nintendo, Electronic Arts, Sony dominate global sales.

Trends: Sales peaked in early 1990s, steadily declined post-2000.

Predictive Frameworks

Decision Tree (Gini Index):

Accuracy = 96.9%

Key predictor: NA_Sales threshold at 0.26.

Random Forest (Champion):

Accuracy = 99.2%, Sensitivity = 99.6%, Specificity = 97.8%.

Most influential predictors: NA_Sales, JP_Sales, EU_Sales.

Neural Network (Challenger):

Accuracy = 99.8%, but computationally intensive and less interpretable.

Logistic Regression:

Accuracy = 99.8%, interpretable baseline with strong predictive power.

Time Series Framework

ARIMA (2,1,0):

Built on monthly average sales.

ADF test confirmed stationarity (p = 0.017).

RMSE = 0.75, MAE = 0.40.

Forecast suggests modest growth, but wide confidence intervals → high uncertainty.

📊 Key Insights

North America is the strongest predictor of global success.

Games with NA_Sales > 0.26M typically succeed globally.

Low regional sales predict failure.

Without traction in at least one major region, global underperformance is likely.

Random Forest is the most reliable predictive model.

Balances accuracy, sensitivity, and interpretability.

Uncertainty remains high.

ARIMA forecasts show possible growth but wide confidence bands.

🚀 Recommendations

Prioritize North American launches for new titles; test product-market fit here first.

Adopt Random Forest as the primary prediction tool for pre-launch evaluations.

Use inline sales thresholds (NA > 0.26M, Global > 0.6M) as early performance KPIs.

Plan for uncertainty by preparing base, best, and worst-case scenarios using ARIMA confidence intervals.

📂 Deliverables

A3 Business Insight Report.pdf – Full written report with analysis, charts, and business insights.

Appendix – R code and model outputs included in the report.

README.md – This documentation.

📜 References

Kaggle. (n.d.). Video Game Sales Dataset. Retrieved from https://www.kaggle.com/

Abi Joshua George. (2025). Assignment A3: Business Insight Report – Video Game Sales Forecasting. Hult International Business School
