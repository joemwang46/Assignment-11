## Model Performance ##

Logistic Regression:
- 230 correct predictions, 19 FP, 11 FN
- Accuracy: 0.8846
- Precision: 0.9127
- Recall: 0.8582
- PnL was good until final year. ended with negative PnL. outperformed Benchmark

Random Forest Classifier:
- 222 correct prediction, 27 FP, 11 FN
- Accuracy:  0.8538
- Precision: 0.9068
- Recall:    0.7985
- PnL was good until final year. ended with negative PnL. under-performed Benchmark

XGB Classifier:
- 213 correct predictions, 30 FP, 17 FN
- Accuracy:  0.8192
- Precision: 0.8595
- Recall:    0.7761
- PnL was good until final year. ended with negative PnL. under-performed Benchmark


## Feature Importance ##

Feature importance varied a lot depending on both model type and ticker. For Logistic Reg, return_1d and sma_5 came out on top for most tickers, for Random Forest it varied a lot, but for XGB, sma_5 was consistently the most important. 

## Overal performance ##

Overall, Logistic Regression performed the best. It had best predictive accuracy and precision. And it also had the best returns in the backtest. However, I had to increase the iterations to 1000 for convergence.


## Limitations of ML in Finance ##

Machine learning struggles in financial forecasting because markets are noisy, non-stationary, and heavily driven by regime changes that models can’t easily capture. Signals that look predictive in-sample often disappear out-of-sample due to overfitting, data-snooping bias, and feedback effects from other market participants exploiting the same patterns. Many true risk factors have low signal-to-noise ratios, so ML models can mistake randomness for structure. Finally, ML assumes stable relationships, but financial markets frequently break those assumptions—especially during stress events when predictions matter most.

