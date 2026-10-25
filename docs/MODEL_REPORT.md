# PulseCheck AI: Machine Learning Model Report

## 1. Objective

PulseCheck AI explores the use of unsupervised anomaly detection to identify unusual patterns in CPU and memory utilization.

The project combines time-series feature engineering, an Isolation Forest model, threshold selection, and a live system monitoring interface.

The objective is to flag patterns that merit investigation, not to diagnose the underlying cause of a system issue.

## 2. Dataset

The training workflow used the OS-Metrics: Performance Time Series (PTS) Dataset for Anomaly Detection and Change Point Detection.

**Dataset DOI:** https://doi.org/10.5281/zenodo.17981352

The CPU and memory time series each contained 40,320 observations sampled at 15-second intervals, representing seven days of measurements.

Documented event intervals were used to derive anomaly labels for evaluation.

The combined labeling process identified 4,017 anomalous observations and 36,303 normal observations in the prepared dataset.

These labels reflect documented dataset events. They should not be interpreted as confirmed hardware failures.

## 3. Feature Engineering

The model uses ten CPU and memory features:

1. CPU utilization
2. Memory utilization
3. Change in CPU utilization
4. Change in memory utilization
5. One-minute rolling mean of CPU utilization
6. One-minute rolling mean of memory utilization
7. Five-minute rolling mean of CPU utilization
8. Five-minute rolling mean of memory utilization
9. Five-minute rolling standard deviation of CPU utilization
10. Five-minute rolling standard deviation of memory utilization

The rolling windows use a 15-second sampling interval.

- One minute corresponds to four samples.
- Five minutes corresponds to twenty samples.

These features capture both current resource utilization and changes in recent behavior.

## 4. Model Architecture

The saved model is a Scikit-learn pipeline consisting of:

- Median imputation using `SimpleImputer`
- `IsolationForest` with 200 estimators

The Isolation Forest was trained on normal observations from the training partition.

The training workflow used a chronological split to preserve time order:

- 60% training partition
- 20% validation partition
- 20% held-out test partition

The validation partition was used to select the anomaly detection threshold.

The full training workflow is documented in [`PulseCheck_AI_Model_Training_Notebook.ipynb`](../PulseCheck_AI_Model_Training_Notebook.ipynb).

## 5. Threshold Selection

The model's anomaly score is calculated by negating the output of `decision_function`.

An observation is flagged as anomalous when:

`anomaly_score >= 0.10000384221612314`

The application displays this threshold rounded to `0.1000`.

Threshold selection targeted validation precision of at least 80%, seeking a balance between false alarms and missed anomalies.

The score is a model-derived anomaly score, not a calibrated probability.

## 6. Evaluation Results

The held-out test partition contained:

- 7,400 normal observations
- 664 anomalous observations

The final evaluation produced the following confusion matrix:

| | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Actual normal | 7,319 | 81 |
| Actual anomaly | 126 | 538 |

Reported metrics:

| Metric | Result |
|---|---:|
| Accuracy | Approximately 97.0% |
| Precision | Approximately 86.9% |
| Recall | Approximately 81.0% |
| F1 score | Approximately 83.9% |

Precision indicates the proportion of flagged observations that were anomalous according to the dataset labels. Recall indicates the proportion of labeled anomalous observations detected by the model.

These measurements describe performance on this particular held-out test set. They do not establish equivalent performance on unseen devices or all real-world conditions.

## 7. Deployment

The trained model and configuration are saved separately from the dashboard.

The Streamlit application:

1. Collects current system metrics.
2. Derives CPU and memory features.
3. Applies the saved preprocessing and model pipeline.
4. Calculates the anomaly score.
5. Compares the score with the configured threshold.
6. Displays the result alongside resource usage and investigation guidance.

Disk utilization is displayed independently and is not one of the ten model input features.

## 8. Limitations

### Domain shift

The model was trained on a separate system performance dataset. The distribution of CPU and memory usage on an individual laptop may differ from the training data.

### Event labels

The labels are derived from documented event intervals. They represent dataset annotations, not independently verified hardware faults.


### Startup history

Rolling statistics are more representative after enough observations have been collected. Short startup windows can produce less stable features than full five-minute windows.

### Interpretability

The model identifies unusual patterns but does not establish causation. Resource-specific guidance is intended to help users investigate, not to claim that a particular resource caused an alert.

## 9. Conclusion

PulseCheck AI demonstrates an end-to-end machine learning workflow: time-series preparation, feature engineering, unsupervised anomaly detection, validation-based threshold selection, held-out evaluation, and integration into a live monitoring dashboard.

A key next step is evaluating the model on representative data from additional machines and calibrating it for individual device behavior.