# Crude-Oil-Price-Prediction
Crude Oil Price Prediction
Forecasting Crude Oil Prices

I.	Overview
Fuel represents one of the largest operating costs for airlines, and crude oil price volatility directly impacts jet fuel expenses. This project develops a time series forecasting solution to anticipate short- to mid-term crude oil prices (1–3 months), enabling the finance and operations team to improve budget planning, purchasing strategies, and risk management. The results are presented in an interactive Power BI dashboard.

II.	Objectives

-	Make forecast for crude oil prices.
-	Provide interpretable forecasts to support decision-making.
-	Deliver clear visualizations via Power BI with automatic refresh capabilities.

III.	Approach

1.	Data Pipeline

-	Sourced historical crude oil prices WTI from public Yahoo Finance since 2000.
-	Cleaned and preprocessed data (resampling, missing value handling, normalization).
-	Perform EDA and decide relevant features
-	Exported datasets in CSV formats for model input and Power BI integration.

2.	Forecasting Model

3.	Tested models: 
-	Evaluated performance using MAE and R^2 and Directional Accuracy
-	Generated forecasts for 1–3 months ahead. Also depend on time granularity.

4.	Visualization & Reporting
Designed Power BI dashboards with:
-	Actual vs. forecasted crude oil prices (time-series view).
-	Slicer for models and time granularity.


IV.	Results Summary

-	The models achieved fairly good accuracy (low MAE/RMSE).
-	Forecasts aligned well with short-term market movements, providing actionable insight.
-	Finance users can quickly interpret trends and compare forecasts vs. actuals directly in Power BI.



V.	Constraints & Scope

-	Focused on **crude oil prices**, not jet fuel directly (due to public data availability).
-	Time horizon limited to **short-/mid-term forecasts** (1–3 months).
	Used only **public, non-sensitive datasets**.


VI.	File instructions

-	EDA_Cleaned_Notebook.ipynb to perform and extract relevant features and historical dataset.
-	Oil_Price_Prediction_Notebook.ipynb to perform some data engineering, train, test the models and generating forecasting and testing results based on time granularity. 
-	Merge_csv_files.ipynb to merge the forecasting results together for Power BI


VII.	Key Takeaways
-	Airlines can leverage forecasting to anticipate cost fluctuations and plan fuel strategies.
-	The pipeline is lightweight and reproducible, making it easy to extend with other models or external indicators.
-	The dashboard bridges technical insights and business decisions, empowering non-technical users.

IX.  Instruction to refresh data on Power BI
- Step 1: Run every cells in the EDA notebook to get the relevant_features.csv file.
- Step 2: Run Oil_price_prediction_notebook for all time granulity(Rerun EVERY cells for each time granulity) to get the csv file for each time granulity.
- Step 3: Run Merge_csv_files on the csv files obtained from step 2 to get the combined data. Then save that on github or a preferred website.
- Step 4: Open the Power BI file, go to Transform Data under the Home tab. On the opened window, select the combined_oil_price_data query. Select Source under Query setting on the right hand side, you should see a Power Query M line in the middle section of the screen. Change the link in that line to the link from step 3. Hit the tick mark on the left to run it.
- Step 5: Select the drop down menu on Column2, uncheck Backtest Prediction(may need to click "Load More" to see it), then hit OK.
- Step 6: Select Close and Apply under the home tab of the window.
