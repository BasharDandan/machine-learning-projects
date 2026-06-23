## 🛠️ Tabular Machine Learning Projects

### Project Description
An introductory machine learning workspace exploring data cleansing pipelines, deterministic array distribution, ensemble modeling, and hyperparameter grid cross-validation. This project establishes the baseline analytical workflow required to prepare raw structures, isolate metrics, and optimize estimators before transitioning into deep neural architectures.

### Problem Statement
Implementing an optimized ensemble machine learning pipeline capable of scanning structured tabular housing attributes and predicting continuous market values. The core challenge is to mitigate data leakage and feature bias through strategic shuffling, evaluate performance across multiple continuous error constraints, and isolate the absolute optimal model configuration using brute-force hyperparameter space exploration.

### Workflow & Feature Extraction
* **Deterministic Data Shuffling:** Leveraging Pandas data shuffling techniques with explicit random seeds to break sequential bias, randomize data distribution across row dimensions, and ensure robust split layouts before partitioning data.
* **Continuous Error Auditing:** Implementing strict validation baselines by evaluating model residuals using both Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) to monitor prediction shifts in absolute dollar terms and relative error scales.

### Technologies Used
* Python
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib

### Model Architecture
The tabular estimation pipeline evaluates a progressive series of ensemble structures:
* **Baseline Regressor:** A default Decision Tree structure serving as the architectural performance floor.
* **Random Forest Ensemble:** A robust meta-estimator (`RandomForestRegressor`) that fits a multitude of parallel decision trees on various sub-samples of the dataset and uses averaging to control over-fitting.
* **GridSearchCV Optimization Engine:** A comprehensive grid search wrapper (`GridSearchCV`) executing exhaustive parameter tuning across a pre-defined hyperparameter configuration matrix (evaluating tree depth bounds, minimum split thresholds, and estimator count constraints).

### Results
Systematic hyperparameter search and data randomization unlocked massive performance gains, stabilizing variance errors and optimizing estimation bounds:
* **Hyperparameter Convergence:** The grid-searched Random Forest model located the optimal parameter intersection, dramatically driving down validation MAE and MAPE curves compared to un-tuned baselines.
* **Generalization Bounds:** Proven that combining deterministic shuffling with ensemble grid arrays completely suppressed model overfitting, ensuring high accuracy stability on unseen testing rows.

### Sample Predictions
Evaluation dataframes displaying real property values directly alongside grid-optimized model predictions, computing local percentage error residuals for each sample row to verify overall reliability.

### Future Improvements
* Transition from standard Random Forests to modern gradient-boosting frameworks like XGBoost or LightGBM to improve training speeds and prediction accuracy.
* Implement feature engineering pipelines to automate categorical one-hot encoding and target-encode high-cardinality columns.
* Scale up the pipeline optimization scope by integrating random search cross-validation (`RandomizedSearchCV`) to process high-dimensional hyperparameter grids with less computational overhead.
