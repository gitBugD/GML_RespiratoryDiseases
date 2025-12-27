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

### Conclusion

Now, we can see that `Random Forest` `XGBoost` use a lot `Surface area (sq. km)` for their predictions. This value is also correlated with the country and doesn't bring clear information. It may be interesting to replace this with the ratio `population per sq. km` because the density of population tells us more about the people environment. 

## Suppression of total variables

In this part, we deleted `GDP (current US$)`, `Surface area (sq. km)`, `Total area (Square Km)` and `Population, total` because they added bias in our models and modified `Sulphur oxides (tonnes)` and `Total sales of agricultural pesticides (tonnes)` by dividing them with the number of population to get a ratio. We obtained better metrics and more logical results.

### Conufsion matrix
![1021_prev_confusion_matrix](./outputs_prev/1021_prev_conf_matrix_v1_2.png)

### Metrics
![1021 prev metrics](./outputs_prev/1021_prev_metrics_v1_2.png)

### Best features
![1021 prev best features](./outputs_prev/1021_prev_best_features_v1_2.png)

### Conclusion

Because XGBoost seems to love biased variables, we looked at the values of t2m and others temperatures variables, and we saw that these values have a lot of precision. For exemple, one of them is `298.236619`. Six digits after comma is useless. We will round them at one value after comma to follow the meteorological standard. We will also delete `d2m` and `skt` because they have a correlation of almost 1 with `t2m`. The correlation is logic they all are temperature measurement and having 3 times the same col gives just overfitting. 

## Modifying temperatures

Even after rounding them to 1 digit after comma, we got almost 350 unique temperatures. It was too much and this also brought bias. So we diced to roung them directly to the unit. We obtained the following results.

### Conufsion matrix
![1021_prev_confusion_matrix](./outputs_prev/1021_prev_conf_matrix_v1_3.png)

### Metrics
![1021 prev metrics](./outputs_prev/1021_prev_metrics_v1_3.png)

### Best features
![1021 prev best features](./outputs_prev/1021_prev_best_features_v1_3.png)

### Conclusion

Now, we see that we don't have the same col multiple times.

## u10 problem

The ´u10´ variable is the vector on the west-east axis of the wind speed. A positive value means that the direction is from west to east and vice-versa. The problem with it is that it has a very high correlation of 0.74 with the target. This high correlation should be analyzed and explained. Indeed, it is highly unlikely that the wind direction itself has such a strong direct biological impact on the disease. Instead, this variable acts as a geographical proxy (or confounding variable). Positive values (Westerlies) predominantly characterize temperate zones in the Northern Hemisphere (e.g., Europe, North America), which tend to have aging populations and higher diagnosis rates. Conversely, negative values (Trade Winds) characterize tropical regions. Thus, the model is likely using ´u10´ to implicitly identify the latitude or development level of a country rather than evaluating the genuine physical effect of wind on pollution dispersion.

Conversely, the ´v10´ variable (North-South component) shows much lower importance. This is physically consistent with global atmospheric circulation patterns, which are predominantly zonal (East-West) due to the Earth's rotation. While ´u10´ creates distinct, stable bands separating tropical from temperate regions (and thus correlates with socio-economic factors), ´v10´ is highly variable and often averages out close to zero over a year. Consequently, it fails to act as a distinct geographic identifier for the model.

![correlation scatter plot](./outputs_prev/corr_scatter.png)

### Conclusion

To resolve this bias, we may want to combine ´u10´ and ´v10´ to also have the wind speed. To implement this new column, for each line, we will calculate $\sqrt{u10^2+v10^2}$.


## 9019 dataset with 100 different 80/20 splits

As proposed by Hector during the presentation, we executed 100 different 80/20 splits to have a larger view on the results. In the graphs below, we can see that there is no signifcant standard deviation.

![metrics](./outputs_prev/9019_prev_precise_metrics_100.png)

## More models and MLP

We tested a custom MLP and various forest algorithms, all of which yielded excellent results. Interestingly, every model highlighted development indices as key predictors, suggesting a direct correlation between COPD and a country's development level. To refine our analysis, our next step involves creating a dataset with more comprehensive data on smoking habits.

![metrics](./outputs_prev/9019_prev_metrics_mlp.png)

![best features](./outputs_prev/9019_prev_best_features_mlp.png)


## New dataset

Afer getting the previous results, we wanted to add more specific datas about the COPD. By reading the [CHUV's article on COPD](https://www.chuv.ch/fr/offre-en-soins/maladies/maladie/web_mld_321/bpco-bronchopneumopathie-chronique-obstructive), we learn that COPD is a group of diseases that cause an "airway blockage and breathing problems". We also learn that BPCO can be caused by tobacco, asthma, exposure to air polluants, genetic factors and respiratory infections. With all that new knowledge, we wanted to analyze the impcat of these indicators, compared to our initial indicators, on the prediction of our models. Before, the tobacco use was not used by our models because it has too many missing values, so it was droped by our pre-process functions. Here, we forced the pre-process to no drop this indicator and to fill missing values. Also, we specifically added the values :
- medical practitioners (to know if there is a problem of diagnostic)
- % of population age 65+ (because BPCO takes years to develop and is more present on older people. [BPCO : broncho-pneumopathie chronique obstructive](https://www.chu-lyon.fr/bpco-broncho-pneumopathie-chronique-obstructive))
- % of industry workers
- % of agricultural workes (to know if there is a difference between different the type of work)

### Conufsion matrix
![9019_prev_conf_matrix_specialized](./outputs_prev/9019_prev_conf_matrix_specialized.png)

### Metrics
![9019_prev_metrics_specialized](./outputs_prev/9019_prev_metrics_specialized.png)

### Best features
![1021 prev best features](./outputs_prev/9019_prev_best_features_specialized.png)

### Conclusion
We can see that our best F1-score dropped by 0,019, which negligeable and indicates that our performances are very close. The more interesting results are on the most used features. We can see that now `Random forest`, `XGBoost` and `CatBoost` have the tobacco use in their top 3 features, which with what the CHUV's page was telling. We can also see the the population density and the poverty count are still amongst the most used features. We will later analyze more precisely the behavior of those values. 

## 40-59 age class
This [article of CHU Lyon](https://www.chu-lyon.fr/bpco-broncho-pneumopathie-chronique-obstructive) suggests that BPCO starts to appear on people after 40 years of age. To analyse better this behavior, we will add the % of people 40-59 and replace the % of people 65+ by the % of people 60+ to have a more general coverage.

### Conufsion matrix
![9019_prev_conf_matrix_specialized](./outputs_prev/9019_prev_conf_matrix_specialized_2.png)

### Metrics
![9019_prev_metrics_specialized](./outputs_prev/9019_prev_metrics_specialized_2.png)

### Best features
![9019_prev_best_features_specialized_2](./outputs_prev/9019_prev_best_features_specialized_2.png)

### Conclusion
In the previous graph, we can observe that we obtained similar scores, which indicates that the models can class the prevalence correctly. Then, if we look more at the most important features, we can see that the best models still give a huge importance to the tobacco use and that lightGBM uses a lot the % of people 60+ and the physicians (doctors) per 1000 people. We also see that people 40-59 is not very important.

## Find the best features in most cases
The problem with our implementation is that the notebook chooses only the best features for on random seed. Our goal is to find the best features for most cases and rank them. We created new plots which show the top features with the best average score, the most present features, the average importance per model and importance variability of the best models. We also modified the confusion matrix to show the average results in %. The models tested 10 different splits.

### Conufsion matrix
![9019_prev_conf_matrix_norm_n10](./outputs_prev/9019_prev_conf_matrix_norm_n10.png)

### Metrics
![9019_prev_metrics_norm_n10](./outputs_prev/9019_prev_metrics_norm_n10.png)

### Best features
![9019_prev_best_features_total_n10](./outputs_prev/9019_prev_best_features_total_n10.png)

### Conclusion

Now, we can confirm that our models use mostly the tobacco prevalence, population density, people using basic sanitation services, poverty headcount and physicians per 1000.

## Keep only years with no missing values for tobacco prevalence

The problem with tobacco prevalence was that it had data only for 5 different years (2000, 2005, 2007, 2010 and 2015). So most of the datas were imputed, which means most of them were not correct. Here, we choose to only train our models on years we had data for tobacco. We deleted the years with missing values only after the data imputation so the imputed values are better.

### Conufsion matrix
![9019_prev_conf_matrix_tobacco_n10](./outputs_prev/9019_prev_conf_matrix_tobacco_n10.png)

### Metrics
![9019_prev_metrics_tobacco_n10](./outputs_prev/9019_prev_metrics_tobacco_n10.png)

### Best features
![9019_prev_best_features_tobacco_n10](./outputs_prev/9019_prev_best_features_tobacco_n10.png)

### Conclusion

Firstly, we can see that the F1-score of our best models reduced by 0,1. This seems logic because because the dataset size has been divided by 5 and thus models have less data to train. It is actually reassuring to lose only 0,1 of F1-score by dividing our dataset by 5. It means that our data is robust. Then, if we look at the best features, we can see the population density is still at the top and a new feature appeared : the surface pressure (sp). Then, we still have the same features as before : physicians per 1000 people, tobacco use and people using at least basic sanitation services. Having a strong importance with sp is debatable because it can act as proxy for the country because the surface pression may not change through the years. 

## Updating the data preparation

The problem was that the tobacco data was imputed using a forward fill, which introduced no evolution. To solve this, a robust data pipeline was developed, combining linear temporal interpolation with MICE to generate realistic data for missing years (1990-2019). This hybrid approach prioritizes historical continuity for key risk factors like tobacco use, significantly reducing the noise typically introduced by purely statistical imputation methods.

### Conufsion matrix
![9019_prev_conf_matrix_new_dp_n10](./outputs_prev/9019_prev_conf_matrix_new_dp_n10.png)

### Metrics
![9019_prev_metrics_new_dp_n10](./outputs_prev/9019_prev_metrics_new_dp_n10.png)

### Best features
![9019_prev_best_features_new_dp_n10](./outputs_prev/9019_prev_best_features_new_dp_n10.png)

### Conclusion

We can see that the tobacco use became the most important feature, followed by the other classic best features. We can also see that coal consumption and sulfur emission are pretty important.

## Analyze how the values influence the target

While high predictive accuracy validates our models, the core objective of this study is to understand the drivers of COPD. In this section, we move beyond performance metrics to analyze the directional relationship between environmental factors and disease prevalence. Using SHAP and Partial Dependence Plots, we investigate whether specific variables (such as pollution levels, temperature, or humidity) act as risk factors (positively correlated with prevalence) or protective factors (negatively correlated), effectively mapping the environmental determinants of the disease.

### SVM
![9019_prev_beeswarm_svm_n10](./outputs_prev/9019_prev_beeswarm_svm_n10.png)

### Random Forest
![9019_prev_beeswarm_RF_n10](./outputs_prev/9019_prev_beeswarm_RF_n10.png)

### XGBoost
![9019_prev_beeswarm_XGBoost_n10](./outputs_prev/9019_prev_beeswarm_XGBoost_n10.png)

### LightGBM
![9019_prev_beeswarm_LightBGM_n10](./outputs_prev/9019_prev_beeswarm_LightBGM_n10.png)

### CatBoost
![9019_prev_beeswarm_CatBoost_n10](./outputs_prev/9019_prev_beeswarm_CatBoost_n10.png)

### Conclusion

In the previous graphs, we can clearly see the effect of each feature on the COPD prevalence. We can see that the COPD grows with fewer agriculturals worker, more physicians per 1000 capita, higher GDP per capita, lesser population density, more tobacco consumption, fewer electricity use per capita, higher access to clean fuels and technologies for cooking, fewer coal consumption, less compulsory education duration, higher PM2.5_pollution and lesser u10. 