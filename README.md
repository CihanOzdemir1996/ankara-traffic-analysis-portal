# 🚗 Ankara Traffic: AI Prediction & Deep Statistical Audit Portal

This project is a high-end **Data Science** and **AI Software Engineering** application. It moves beyond simple visualization by implementing a rigorous statistical framework to validate traffic data and predict average vehicle speeds in Ankara.



## 🔬 Scientific Methodology (The 5-Step Audit)

Before any AI modeling occurs, the system subjects the raw data to a comprehensive **Econometric Audit** to ensure the integrity of the results:

### 1. Outlier Removal (IQR Method)
Using the **Interquartile Range (IQR)**, the system identifies and removes noise and anomalies (e.g., GPS errors or extreme speed spikes). This ensures the AI model is trained on realistic traffic patterns.


### 2. Normality Testing (Shapiro-Wilk)
The system evaluates the distribution of speed data using the **Shapiro-Wilk Test**. 
- **Insight:** Based on the $p-value$, the engine dynamically switches between **Pearson** (for normal distributions) and **Spearman Rank** (for skewed distributions) correlation methods.


### 3. Multicollinearity Assessment (VIF)
To ensure model stability, we calculate the **Variance Inflation Factor (VIF)** for independent variables (Density, Weather, Hour). 
- **Goal:** We verify that no features are redundant or overly correlated (VIF < 5), allowing the model to assign accurate importance to each factor.


### 4. Homoscedasticity Check (Breusch-Pagan)
We validate the **Homogeneity of Variance** using the **Breusch-Pagan Test**. This confirms that the model's error rate remains constant across all traffic conditions, preventing biased predictions.


### 5. Autocorrelation Analysis (Durbin-Watson)
The system performs a **Durbin-Watson Diagnostic** to check for independence of residuals, ensuring that time-series patterns do not invalidate the regression assumptions.

---

## 🤖 AI Features & Interactive Simulation
- **AI Prediction Engine:** A multivariate Linear Regression model that forecasts speed based on density, hour, and weather.
- **Dynamic Traffic Simulator:** Real-time "What-If" analysis tool for urban planners.
- **Geospatial Analytics:** Interactive mapping of traffic hotspots across Ankara.
- **Advanced Visualization:** Automated heatmaps and hourly flow dynamics.

## 🛠️ Tech Stack
- **Languages:** Python
- **Backend/AI:** Scikit-Learn, Statsmodels, SciPy
- **Frontend:** Streamlit
- **Data:** Pandas, NumPy
- **Visuals:** Seaborn, Matplotlib

## 👤 Author
Developed by **Cihan Özdemir** [![LinkedIn](https://img.shields.io/badge/LinkedIn-ozdemircihan-blue?logo=linkedin)](https://www.linkedin.com/in/ozdemircihan/)
[![GitHub](https://img.shields.io/badge/GitHub-CihanOzdemir1996-black?logo=github)](https://github.com/CihanOzdemir1996)

---
*Verified with Advanced Statistical Frameworks (VIF, Shapiro-Wilk, Breusch-Pagan).*
