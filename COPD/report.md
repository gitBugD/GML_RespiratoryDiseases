# COPD part

## First Predictions

At first, we took the dataset `1021/COPD_prevalence_rate.csv`, imputed missing values and encoded string values with functions in `data_preparation` file. Then, we wanted to see if a classification would show good results. For this, we separated observations in three equal size classes : `low`, `medium` and `high`. Then, we trained a **logistic regression**, **random forest** and **XGBoost**. Here are the results :

### Conufsion matrix
![1021_prev_confusion_matrix](./outputs_prev/1021_prev_conf_matrix.png)

### Metrics
![1021 prev metrics](./outputs_prev/1021_prev_metrics.png)

### Best features
![1021 prev best features](./outputs_prev/1021_prev_best_features.png)

## Second prediction TODO

Then, we enriched our dataset and added more years to get better results. Here, we used the `9019/COPD_prevalence_rate.csv`.

### Conufsion matrix
![1021_prev_confusion_matrix](./outputs_prev/9019_prev_conf_matrix.png)

### Metrics
![1021 prev metrics](./outputs_prev/9019_prev_metrics.png)

### Best features
![1021 prev best features](./outputs_prev/9019_prev_best_features.png)

### Conclusion for second part

We needed to make the `data_preparation.py` file more generic because it selected specified cols, so it did not work on every dataset. Then, the imputation we defined took 12 minutes on the new dataset which shape was (4950, 37), which was a lot of time. So we decided to keep the `IterativeImputer` but set `max_iter` at **5** and `n_nrearest` at **15** to impute faster. We also forgot to take country code out of the train and test sets, so we implemented this for the next parts.

## 1021 dataset without country and year in train/test

As said before, in this part, we executed our notebook without the counry code and year to get less bias and the metrics were weaker, but results may be better.

### Conufsion matrix
![1021_prev_confusion_matrix](./outputs_prev/1021_prev_conf_matrix_v1_1.png)

### Metrics
![1021 prev metrics](./outputs_prev/1021_prev_metrics_v1_1.png)

### Best features
![1021 prev best features](./outputs_prev/1021_prev_best_features_v1_1.png)