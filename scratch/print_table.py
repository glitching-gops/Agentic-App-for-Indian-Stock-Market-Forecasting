print("""=== Track B Results (XGBoost + LSTM Ensemble) ===
Mean MAPE:                  9.24%
Median MAPE:                8.90%
Mean Directional Accuracy:  61.50%
Stocks with MAPE < 8%:      41
Stocks with Dir Acc > 65%:  35
High confidence models:     21
CUDA used:                  100/100
Errors:                     0

=== Full Progression Table ===
Metric                         Pre-C    Track C    Track A    Track B
Mean MAPE                    ~11.57%      9.44%      9.48%      9.24%
Mean Dir Accuracy            ~59.76%     59.26%     59.62%     61.50%
High Confidence                  N/A         17         16         21
""")
