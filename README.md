# Traffic Accident Severity Prediction

## Project Overview

This project builds a model to predict the severity of traffic accidents using data from the Traffic Accident Dataset on Kaggle, containing US traffic data (2016-2023). Since Montreal-specific data is unavailable, we selected US cities with similar characteristics (population, road layouts, climate, etc.).

- **Target City**: Montreal, QC
- **Training Cities**: Philadelphia, Minneapolis, Chicago, Pittsburgh, Buffalo, Cleveland, Seattle, Detroit, Milwaukee, Rochester, Denver, Albany, Portland
- **Test City**: Boston, MA (due to its similarity to Montreal)

## Dataset Information

The dataset initially consists of 260,269 entries with 32 features (numeric, categorical, and boolean variables). We performed extensive preprocessing to ensure the data's quality.

### Key Columns:
- **Severity** (Target variable)
- **Weather-related**: Temperature(F), Wind_Speed(mph), Precipitation(in)
- **Traffic conditions**: Crossing, Traffic_Signal, Is_Highway, Is_Night
- **Time features**: Start_Time, End_Time, Hour, Month

## ETL and Data Preprocessing

1. Filtered data for selected cities.
2. Handled missing values and fixed data types.
3. Addressed outliers and unrealistic values:
   - Max distance of 254.4 miles, temperatures up to 189°F, and wind speeds of 254.3 mph were corrected.
4. Created additional features (e.g., Is_Weekend, Is_Dry_Weather).

## Exploratory Data Analysis (EDA)

Studied distributions and correlations between features and severity. Identified that weather conditions and road types significantly affect accident severity.

## Modeling Approach

### Preprocessing:
- Applied Borderline-SMOTE to address class imbalance.
- Conducted hyperparameter tuning using GridSearch.

## Results

We evaluated various models and resampling techniques to predict traffic accident severity. Below is a summary of the key metrics:

| **Model**                               | **Train Accuracy** | **Train F1 Score** | **Train Kappa** | **Test Accuracy** | **Test F1 Score** | **Test Kappa** |
|-----------------------------------------|--------------------|--------------------|-----------------|-------------------|-------------------|----------------|
| Decision Tree (Baseline)                | 0.998              | 0.998              | 0.996           | 0.725             | 0.726             | 0.408          |
| XGBoost                                 | 0.889              | 0.887              | 0.754           | 0.802             | 0.795             | 0.557          |
| XGBoost + Class-wise SMOTE              | 0.907              | 0.908              | 0.803           | 0.795             | 0.794             | 0.561          |
| XGBoost + Borderline-SMOTE              | 0.865              | 0.866              | 0.714           | 0.794             | 0.794             | 0.561          |
| LightGBM + Borderline-SMOTE              | 0.852              | 0.853              | 0.685           | 0.798             | 0.796             | 0.566          |
| **Final Model (Full Dataset) + test on Boston data**          | 0.793              | 0.796              | 0.565           | **0.705**         | **0.716**       | **0.403**      |

- **Baseline Decision Tree**: Showed significant overfitting with a large gap between train and test performance.
- **XGBoost**: Reduced overfitting and achieved more balanced performance, generalizing better to unseen data.
- **XGBoost with SMOTE (Class-wise and Borderline)**: Improved recall for minority classes, with overall performance comparable to base XGBoost.
- **Final Model on Full Dataset**: Stable results across all metrics.
- **Performance on Unseen Data (Boston, MA)**: Moderate drop in performance on new data, indicating potential challenges in generalizing to other regions.

## Conclusion

The XGBoost model with Borderline-SMOTE resampling offered the best balance between precision and recall, addressing the class imbalance issue effectively. While the model performed reasonably well on unseen data from Boston, MA, it showed signs of slight underperformance, suggesting that additional tuning or region-specific data may further improve performance.

## Future Work

1. **Data Augmentation**: Collect additional data from different regions to improve generalization.
2. **Feature Engineering**: Explore additional features to enhance model performance.
3. **Model Tuning**: Further hyperparameter tuning and experimentation with other models.
