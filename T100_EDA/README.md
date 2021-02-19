# T100: Perform EDA on imbalance data

## Analysis Goal
Perform Exploratory Data Analysis (EDA) on the given dataset.

## Conclusion
This task is done in a team of 2 students.As per the exploratory data analysis (EDA), the data had 30 numerical independent features and 1 categorical target feature.There are no null values and the data is very imbalanced so we have to apply stratified sampling at later stages.During visualization most of the features were negligibly corelated. T1, T3, T7, T10 T12, T14 T16, T17 & T18 are Negatively Correlated features smmaller than -0.1. T4 & T11 are Positively Correlated features greater than 0.1. Also, by the end of the analysis we applied univariate and multivariate analysis.
Added a new function 'scatter_3d' to call 3 features from the whole column list, to print the 3D Scatter plot for random 200 samples for each class 0 & 1.
Also, added 'Pandas Profiling Report EDA .html' file. By the 3D scatter plot we also noticed that the 'Class 0' & 'Class 1' are not linearly separable.
As per analysis we can recoomend to focus on features that came up through the correlation. Also, since most of the features are less correlated, there is a need of feature engineering.
