1. Introduction

Time series forecasting plays a critical role in many real-world applications such as financial analysis, demand forecasting, and sensor monitoring. Traditional statistical models have been widely used for this purpose, but recent advances in deep learning have shown strong potential in capturing complex, non-linear temporal dependencies.

The objective of this project is to implement, compare, and analyze different time series forecasting approaches, ranging from classical statistical models to modern deep learning models with attention mechanisms. The focus of this work is not only on predictive performance, but also on workflow correctness, model comparison, interpretability, and reproducibility, which are essential aspects of production-quality machine learning systems.

2. Dataset Description

For this study, I used Netflix (NFLX) historical closing stock price data, downloaded programmatically using the yfinance library. This dataset represents a real-world financial time series and naturally contains:

Long-term trends

Short-term fluctuations

Noise and volatility

Implicit weekly seasonality due to trading days

Only the closing price was used for forecasting. Missing values were handled using forward fill, which is a common and reasonable assumption for financial time series where trading does not occur every day.

3. Data Preprocessing and Feature Engineering

A key part of this project was designing a consistent preprocessing pipeline that is applied before training any model.

3.1 Feature Engineering

The following features were created:

Lag_1: the previous day’s closing price, capturing short-term dependency

Day of Week: numeric representation of the trading day (0–4), capturing weekly patterns

These features were engineered before model training, ensuring that all models benefit from the same information and that the workflow remains logically correct.

After feature creation, rows containing missing values due to lag computation were removed.

3.2 Scaling and Train–Test Split

The closing price values were scaled using MinMax scaling to bring them into the range [0, 1], which is particularly important for neural network training.

The dataset was then split using a time-based split:

80% for training

20% for testing

Shuffling was intentionally avoided to preserve the temporal structure of the data.

4. Models Implemented

Three distinct models were implemented to represent increasing levels of modeling complexity.

4.1 SARIMA (Statistical Baseline)

SARIMA was chosen as the traditional baseline model due to its ability to explicitly model trend and seasonality in time series data. The model was trained only on the training portion of the data and used to forecast the test period.

This model serves as a reference point to evaluate whether deep learning approaches provide meaningful improvements over classical statistical methods.

4.2 Baseline Deep Learning Model: LSTM

A simple Long Short-Term Memory (LSTM) network was implemented as the baseline deep learning model.

The time series was transformed into fixed-length sequences using a sliding window approach:

Input: previous 60 days

Output: next day’s closing price

This allows the LSTM to learn temporal dependencies beyond what is captured by traditional models.

4.3 Hyperparameter Optimization Using Optuna

To ensure a principled approach to model optimization, Optuna was used for hyperparameter tuning of the LSTM model. The following parameters were optimized:

Number of LSTM units

Batch size

The objective function minimized Mean Absolute Error (MAE) on the validation set. After optimization, the LSTM model was retrained using the best-found parameters.

Hyperparameter tuning was intentionally limited to the baseline LSTM model to maintain experimental clarity and avoid unnecessary complexity.

4.4 Advanced Model: Attention-Based LSTM

The advanced model extends the LSTM by incorporating a temporal attention mechanism. Instead of relying only on the final hidden state, the attention mechanism learns to assign different importance weights to different time steps in the input sequence.

This model is an attention-augmented recurrent neural network, not a Transformer encoder. The design choice was intentional, as it provides improved interpretability while remaining conceptually aligned with the baseline LSTM.

5. Evaluation Metrics

All models were evaluated on the same test set using the following metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

Using multiple metrics allows a balanced understanding of both absolute and relative prediction errors.

6. Results and Model Comparison

A final comparison table was generated to evaluate all models fairly. Each model was trained once, predictions were generated once, and metrics were computed without re-running or overwriting results.

Overall observations:

SARIMA provides a strong statistical baseline.

The LSTM captures non-linear temporal patterns more effectively.

The attention-based LSTM offers similar or slightly improved performance while providing better interpretability.

7. Visualization and Interpretability
7.1 Actual vs Predicted Values

A plot comparing actual values with LSTM and attention-based predictions shows how closely each model tracks real price movements. This visualization helps identify lag, smoothing effects, and general trend alignment.

7.2 Training Loss Curves

Training loss plots for the LSTM and attention models demonstrate stable convergence, indicating that the models learned meaningful patterns without unstable training behavior.

7.3 Attention Weight Analysis

Attention weights were visualized across time steps. The plot shows that recent time steps receive higher attention weights, while distant past values receive lower weights.

This behavior aligns with financial intuition, where recent market movements tend to have a stronger influence on short-term price prediction. This visualization highlights the interpretability advantage of attention mechanisms over standard recurrent models.

8. Conclusion

This project demonstrates a structured and fair comparison of time series forecasting models, progressing from classical statistical methods to modern deep learning approaches with attention mechanisms.

Key takeaways:

Deep learning models can capture complex temporal patterns beyond traditional methods.

Attention mechanisms enhance interpretability by revealing which historical time steps influence predictions.

Correct workflow design, preprocessing, and evaluation are as important as model complexity.

Overall, this work emphasizes practical machine learning principles, reproducibility, and interpretability rather than focusing solely on maximizing predictive accuracy.
