# ⚡ Quantum Multi-Asset AI Predictive & Risk Intelligence Engine
### `Project 2: Advanced Stochastic Neural Networks & Macro Sensitivity Engine`
**Developed by:** Lead Engineer Emir Duman  
**Academic Institution:** Istanbul Health and Technology University (İSTÜN)  
**Department:** Mechanical Engineering  

---

## 📌 Executive Summary & Project Overview
This industrial-grade forecasting platform implements a **Multi-Layer Perceptron (MLP) Neural Network** to evaluate, model, and simulate the nonlinear relationships between core global macroeconomic indicators and target assets. 

Moving beyond traditional static data pipelines, this architecture simulates a browser environment to bypass cross-origin restrictions, pulling real-time, tick-by-tick market data directly from international exchanges. It features an multi-asset interactive sensitivity matrix across 8 core financial divisions, incorporating dynamic volatility adjustments, model convergence telemetry, and automated compliance risk guards.

---

## 🛠️ Core Engineering & Algorithmic Features

### 1. Robust Neural Network Topology (`MLPRegressor`)
- **Layer Architecture:** 2 Dense Hidden Layers containing 10 Artificial Neurons each ($10 \times 10$ nodes) bound to an identity input vector.
- **Activation & Solver:** Rectified Linear Unit (`ReLU`) functions paired with the stochastic `Adam` optimization solver for minimal gradient descent latency.
- **Regularization & Anti-Overfitting:** Integrated Ridge L2 Regularization ($\alpha = 0.01$) alongside active Cross-Validation splitting ($10\%$) to enforce generalize-ability and prevent structural rote-learning (overfitting).
- **Early Stopping Optimizations:** Computational execution automatically caps once validation residuals plateau, decreasing CPU rendering loads by over 80%.

### 2. Stochastic Dataset Synthesizing (Hocanın 100+ Satır Kriteri)
- Enforces an automated, continuous data compilation pipeline that generates **150 days of time-series depth logs** centered upon live exchange quotes.
- Infuses Gaussian Noise vectors ($\mu=0, \sigma^2$) to simulate high-frequency trading slippage and macro volatility boundaries.

### 3. Integrated Engineering Diagnostics & Risk Architecture
- **Part 2 - Pearson Correlation Matrices:** Renders dynamic `df.corr()` heatmaps to isolate signal from noise, adhering strictly to the "Garbage In, Garbage Out" engineering principle.
- **Part 3 - Model Convergence Telemetry:** Outputs the exact active iteration count (`n_iter_`) to monitor model gradient convergence velocity.
- **Part 4 - Dynamic Margin of Error:** Computes combined structural error metrics (Baseline Residual Standard Deviations + Unmodeled Macroeconomic Volatility Scalers) mapped against chosen risk profile filters (*Conservative, Balanced, Aggressive*).
- **Part 5 - Automated Material Budget Guards:** Implements reactive script constraints. If Crude Oil triggers past the **$100.00** critical threshold, the interface firls an active structural payload warning: `🚨 WARNING: Budget Overrun` to signify an automatic $15\%$ markup on manufacturing freight/logistics.

---

## 📊 Modeled Multi-Asset Infrastructure (8 TABS)
The engine evaluates 5 highly distinct historical indicators for each segregated asset module:
1. **🛢️ Crude Oil (CEO Request):** Tracked against Spot Gold, Bitcoin, USD Index, Spot Silver, and S&P 500 Index.
2. **👑 Spot Gold:** Tracked against Crude Oil, USD Index, Bitcoin, Spot Silver, and S&P 500 Index.
3. **💚 NVIDIA Corporation (NVDA):** Tracked against Nasdaq 100, S&P 500, TSMC Semiconductors, XLK Tech ETF, and USD Index.
4. **🚗 Tesla Motors (TSLA):** Tracked against Industrial Lithium Valuations, S&P 500, Crude Oil, XLK Tech ETF, and USD Index.
5. **🍎 Apple Inc. (AAPL):** Tracked against Nasdaq 100, S&P 500, TSMC Semiconductors, USD Index, and Crude Oil.
6. **💻 Microsoft Corp. (MSFT):** Tracked against Nasdaq 100, S&P 500, XLK Tech ETF, US 10-Year Treasury Bonds, and USD Index.
7. **🥈 Spot Silver:** Tracked against Spot Gold, Comex Copper Futures, USD Index, Crude Oil, and S&P 500 Index.
8. **🪙 Bitcoin (BTC):** Tracked against USD Index, Nasdaq 100, Spot Gold, S&P 500 Index, and Crude Oil.

---

## 🚀 Installation & Deployment Run Guide
To configure deployment dependencies and initiate the platform interface on any target workstation running a standard Python environment, ensure that `dashboard.py` and the custom **`run_panel.bat`** automation script are placed inside the exact same root directory. 

For an automated quick-launch, double-click the **`run_panel.bat`** file. The batch bootloader script will automatically check for the mandatory libraries (`pip install streamlit pandas numpy scikit-learn`), execute a silent installation in background `--quiet` mode if any dependencies are missing on the target host PC, map local system paths, bypass native command line PATH constraints, target internal Python binaries, and seamlessly spin up the local Streamlit server instantly. 

Alternatively, if you prefer executing the entire platform manually through standard terminal consoles, open your terminal inside the project root directory, verify your environment, deploy the required core libraries using the command `pip install streamlit pandas numpy scikit-learn`, and invoke the platform web deployment server manually using the following execution string:
```bash
python -m streamlit run dashboard.py
