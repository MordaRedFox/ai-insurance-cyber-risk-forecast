<div align="center">

# 🛡️ AI Insurance Cyber Risk Forecast

### Forecasting financial losses from cyber incidents based on a multi-head neural network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Сменить язык: [Русский](README.md)

</div>

---

## 📖 About the project

<div align="justify">

**AI Insurance Cyber Risk Forecast** – a research project dedicated to developing a system for forecasting financial losses from cyber incidents for insurance companies. The project implements a complete machine learning pipeline: from generating synthetic data and preprocessing it to training a multi-head neural network and visualizing underwriting results.

The key feature is the **multi-head architecture** with a shared backbone and three specialized heads, each solving its own task: estimating the probability of an incident, forecasting conditional and unconditional damage. The model is trained on realistic synthetic data generated taking into account industry specifics, correlations between features, and nonlinear interactions.

The project was carried out as part of research work for the economic conference **"Damage Assessment from Digital Fraud"**.

</div>

---

## ✨ Key features

<div align="center">

<table>
    <tr>
        <td valign="top" width="50%">
            <h3 align="center">🧠 Multi-head neural network</h3>
            <p align="center">
                <img src="https://img.shields.io/badge/🎯-Incident%20probability-blue" alt="Probability">
                <img src="https://img.shields.io/badge/💸-Damage%20in%20case%20of%20incident-red" alt="Damage">
                <img src="https://img.shields.io/badge/📊-Expected%20damage-green" alt="Expected damage">
            </p>
            <div style="text-align: left; margin-left: 20px;">
                <ul style="text-align: left; padding-left: 20px;">
                    <li>🎯 Head T1: cyber incident probability (Binary Cross-Entropy)</li>
                    <li>💸 Head T2: conditional damage in case of incident (MSE)</li>
                    <li>📊 Head T3: unconditional expected damage (MSE)</li>
                    <li>⚙️ Shared backbone with BatchNorm and Dropout</li>
                    <li>🛡️ Gradient clipping for stable training</li>
                </ul>
            </div>
        </td>
        <td valign="top" width="50%">
            <h3 align="center">🖥️ Console application</h3>
            <p align="center">
                <img src="https://img.shields.io/badge/🎮-Interactive%20menu-purple" alt="Menu">
                <img src="https://img.shields.io/badge/📊-Visualization-orange" alt="Visualization">
                <img src="https://img.shields.io/badge/🧪-Tests-green" alt="Tests">
            </p>
            <div style="text-align: left; margin-left: 20px;">
                <ul style="text-align: left; padding-left: 20px;">
                    <li>🎮 Convenient menu with 8 commands</li>
                    <li>📊 Generation of plots for presentation (4 types)</li>
                    <li>🧪 Running pytest tests from the interface</li>
                    <li>🔧 Checking project configurations</li>
                    <li>🌐 Cross-platform (Windows/Linux/macOS)</li>
                </ul>
            </div>
        </td>
    </tr>
</table>

</div>

---

## 🛠 Technology stack

<div align="center">

<table>
    <tr align="center">
        <th>Component</th>
        <th>Version</th>
        <th>Purpose</th>
        <th>Badge</th>
    </tr>
    <tr align="center">
        <td><strong>Python</strong></td>
        <td>3.10+</td>
        <td>Main language</td>
        <td><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></td>
    </tr>
    <tr align="center">
        <td><strong>PyTorch</strong></td>
        <td>2.0+</td>
        <td>Neural networks</td>
        <td><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></td>
    </tr>
    <tr align="center">
        <td><strong>NumPy</strong></td>
        <td>1.24+</td>
        <td>Scientific computing</td>
        <td><img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy"></td>
    </tr>
    <tr align="center">
        <td><strong>Pandas</strong></td>
        <td>2.0+</td>
        <td>Data processing</td>
        <td><img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas"></td>
    </tr>
    <tr align="center">
        <td><strong>SciPy</strong></td>
        <td>1.10+</td>
        <td>Statistics and distributions</td>
        <td><img src="https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?logo=scipy&logoColor=white" alt="SciPy"></td>
    </tr>
    <tr align="center">
        <td><strong>Scikit-learn</strong></td>
        <td>1.3+</td>
        <td>Scaling and metrics</td>
        <td><img src="https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn"></td>
    </tr>
    <tr align="center">
        <td><strong>Matplotlib</strong></td>
        <td>3.7+</td>
        <td>Visualization</td>
        <td><img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?logo=python&logoColor=white" alt="Matplotlib"></td>
    </tr>
    <tr align="center">
        <td><strong>Seaborn</strong></td>
        <td>0.12+</td>
        <td>Statistical visualization</td>
        <td><img src="https://img.shields.io/badge/Seaborn-0.12%2B-3776AB?logo=python&logoColor=white" alt="Seaborn"></td>
    </tr>
    <tr align="center">
        <td><strong>PyYAML</strong></td>
        <td>6.0+</td>
        <td>Configuration management</td>
        <td><img src="https://img.shields.io/badge/PyYAML-6.0%2B-FFD700?logo=yaml&logoColor=black" alt="PyYAML"></td>
    </tr>
    <tr align="center">
        <td><strong>Pytest</strong></td>
        <td>7.0+</td>
        <td>Unit testing</td>
        <td><img src="https://img.shields.io/badge/Pytest-7.0%2B-0A9EDC?logo=pytest&logoColor=white" alt="Pytest"></td>
    </tr>
</table>

</div>

---

## 📊 Dataset documentation

### 📋 Parameters (input features)

<div align="center">

<table>
    <tr align="center">
        <th>No.</th>
        <th>Parameter</th>
        <th>Rationale</th>
        <th>Data type</th>
        <th>Range / Scale</th>
    </tr>
    <tr>
        <td align="center"><strong>1</strong></td>
        <td>Annual revenue (million RUB)</td>
        <td align="left">The higher the revenue, the more assets are at risk and the higher the potential operational disruptions</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>2</strong></td>
        <td>Value of intangible assets (million RUB)</td>
        <td align="left">A direct target of cyberattacks – data and IP, their value correlates with damage</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>3</strong></td>
        <td>Number of employees</td>
        <td align="left">Affects the complexity of access management, the human factor, and response costs</td>
        <td align="center">Discrete</td>
        <td align="center">[1; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>4</strong></td>
        <td>Share of online sales</td>
        <td align="left">The higher the digitalization, the larger the attack surface and dependence on IT infrastructure</td>
        <td align="center">Share</td>
        <td align="center">[0; 1]</td>
    </tr>
    <tr>
        <td align="center"><strong>5</strong></td>
        <td>Cybersecurity expenses over the last year (million RUB)</td>
        <td align="left">Investments in protection reduce risk, but may indicate infrastructure complexity – accounted for as a quadratic effect</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>6</strong></td>
        <td>Number of servers</td>
        <td align="left">More nodes – more potential entry points</td>
        <td align="center">Discrete</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>7</strong></td>
        <td>Share of cloud services in IT infrastructure</td>
        <td align="left">Clouds have their own risks (settings, access), but are often better protected – nonlinear dependence</td>
        <td align="center">Share</td>
        <td align="center">[0; 1]</td>
    </tr>
    <tr>
        <td align="center"><strong>8</strong></td>
        <td>Number of critical business applications</td>
        <td align="left">Failures in them cause maximum financial damage</td>
        <td align="center">Discrete</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>9</strong></td>
        <td>Number of connected IoT/OT devices</td>
        <td align="left">Critical for industrial companies, expands the attack surface</td>
        <td align="center">Discrete</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>10</strong></td>
        <td>Presence of a formal security policy</td>
        <td align="left">Reduces baseline risk</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>11</strong></td>
        <td>Frequency of external pentest</td>
        <td align="left">The frequency of checks correlates with the detection of vulnerabilities</td>
        <td align="center">Ordinal</td>
        <td align="center">0 – less than once a year, 6 – once every six months, 12 – once a year, 24+ – more than a year ago</td>
    </tr>
    <tr>
        <td align="center"><strong>12</strong></td>
        <td>Presence of a SOC / SIEM system and 24/7 monitoring</td>
        <td align="left">Accelerates detection, reduces downtime</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>13</strong></td>
        <td>Percentage of employees who completed cyber hygiene training over the year</td>
        <td align="left">Reduces the risk of phishing and social engineering</td>
        <td align="center">Share</td>
        <td align="center">[0; 1]</td>
    </tr>
    <tr>
        <td align="center"><strong>14</strong></td>
        <td>Presence of an incident response plan (IRP)</td>
        <td align="left">Reduces the cost of eliminating consequences</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>15</strong></td>
        <td>Industry</td>
        <td align="left">Different industries have different frequency and size of damage (finance and healthcare are the most expensive)</td>
        <td align="center">Categorical</td>
        <td align="center">1 – Finance, 2 – Retail, 3 – Manufacturing, 4 – Healthcare, 5 – IT/Telecom, 6 – Other</td>
    </tr>
    <tr>
        <td align="center"><strong>16</strong></td>
        <td>Volume of processed personal data (million records)</td>
        <td align="left">Affects GDPR/152-FZ fines and reputational losses</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>17</strong></td>
        <td>Number of cyber incidents over the last 3 years</td>
        <td align="left">A direct predictor of future losses (risk inertia)</td>
        <td align="center">Discrete</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>18</strong></td>
        <td>Total damage from incidents over the last 3 years (million RUB)</td>
        <td align="left">If present – the best proxy for expected losses</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>19</strong></td>
        <td>Presence of cyber risk insurance coverage in the past</td>
        <td align="left">Shows whether the company was previously underwritten</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>20</strong></td>
        <td>Amount of insurance payouts in the past (million RUB)</td>
        <td align="left">Shows the severity of incurred damage</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>21</strong></td>
        <td>Presence of network segmentation</td>
        <td align="left">Without segmentation, even one compromised computer can lead to total paralysis of the entire network</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>22</strong></td>
        <td>Presence of outdated software/EOL</td>
        <td align="left">Multiplies the probability of a successful attack and completely nullifies the effectiveness of most perimeter defenses</td>
        <td align="center">Binary</td>
        <td align="center">{0, 1}</td>
    </tr>
    <tr>
        <td align="center"><strong>23</strong></td>
        <td>Losses from one hour of downtime (million RUB)</td>
        <td align="left">Is a direct monetization of the business's operational dependence on IT infrastructure</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
    <tr>
        <td align="center"><strong>24</strong></td>
        <td>Maximum allowable downtime of critical services (in hours)</td>
        <td align="left">Exceeding this time entails exponential growth of reputational and contractual penalties</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
    </tr>
</table>

</div>

### 🎯 Target variables

<div align="center">

<table>
    <tr align="center">
        <th>No.</th>
        <th>Parameter</th>
        <th>Rationale</th>
        <th>Data type</th>
        <th>Range / Scale</th>
        <th>Relation to the model</th>
    </tr>
    <tr>
        <td align="center"><strong>Y0</strong></td>
        <td>Actual total damage over the last 12 months (million RUB)</td>
        <td align="left">The root target variable (Ground Truth). This is the figure that comes from insurance statistics</td>
        <td align="center">Continuous</td>
        <td align="center">[0; +∞)</td>
        <td align="left">Used to calculate loss functions (Loss) for heads T1–T3</td>
    </tr>
    <tr>
        <td align="center"><strong>T1</strong></td>
        <td>Probability of an insured event (P_inc)</td>
        <td align="left">Determines whether an incident will occur in the next 12 months. Takes into account nonlinear effects of security maturity (items 10-14) and external factors</td>
        <td align="center">Binary classification / Logit</td>
        <td align="center">[0; 1]</td>
        <td align="left"><strong>Head 1</strong>: Sigmoid (Binary Cross-Entropy)</td>
    </tr>
    <tr>
        <td align="center"><strong>T2</strong></td>
        <td>Conditional damage in case of an incident (Severity)</td>
        <td align="left">Financial losses provided that an attack occurred (excluding zero values)</td>
        <td align="center">Continuous (Regression)</td>
        <td align="center">[0; +∞)</td>
        <td align="left"><strong>Head 2</strong>: Linear output + exponentiation (MSE)</td>
    </tr>
    <tr>
        <td align="center"><strong>T3</strong></td>
        <td>Unconditional expected damage (Pure Premium)</td>
        <td align="left">Calculated as T1 * T2. This is the mathematical expectation of losses, on which the loading is added (expenses for doing business, profit)</td>
        <td align="center">Continuous (Regression)</td>
        <td align="center">[0; +∞)</td>
        <td align="left"><strong>Head 3</strong>: Separate regression for cross-validation</td>
    </tr>
</table>

</div>

---

## 📁 Project structure

```bash
ai-insurance-cyber-risk-forecast/
│
├── cli/
│   ├── commands/                        # Console application commands
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── generation.py
│   │   ├── inference.py
│   │   ├── preprocessing.py
│   │   ├── tests.py
│   │   ├── training.py
│   │   └── visualization.py
│   │
│   ├── services/                        # Auxiliary services of the console application
│   │   ├── __init__.py
│   │   ├── file_finder.py
│   │   ├── folder_opener.py
│   │   ├── test_registry.py
│   │   └── test_runner.py
│   │
│   ├── __init__.py
│   ├── app.py                           # ConsoleApp - main loop
│   ├── menu.py                          # Console application menu
│   └── utils.py                         # Output and launch utilities
│
├── configs/
│   ├── generation_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/
│   ├── presentation/
│   │   └── presentation_companies.csv
│   │
│   ├── processed/
│   │   └── processed_companies.csv
│   │
│   ├── raw/
│   │   └── synthetic_companies.csv
│   │
│   └── visualization_output/
│       ├── 01_probability.png
│       ├── 02_damage.png
│       ├── 03_risk_map.png
│       └── 04_risk_factors.png
│
├── logs/
│
├── models/
│   ├── checkpoints/
│   │   └── best_model.pth
│   └── scaler.pkl
│
├── run_scripts/                        # Process launch scripts
│   ├── __init__.py
│   ├── run_generation.py
│   ├── run_inference.py
│   ├── run_preprocessing.py
│   ├── run_training.py
│   └── run_visualize.py
│
├── src/
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── base.py                     # Base class with common methods
│   │   ├── binary.py                   # Generation of binary features
│   │   ├── continuous.py               # Generation of continuous features
│   │   ├── industry.py                 # Generation of industries
│   │   ├── orchestrator.py             # Coordinator that assembles everything
│   │   ├── pentest.py                  # Pentest generation
│   │   ├── security.py                 # Generation of security expenses
│   │   ├── share.py                    # Generation of share features
│   │   └── target.py                   # Generation of target variable
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base.py                     # Base class with common methods
│   │   ├── encoding.py                 # One-Hot encoding of categories
│   │   ├── feature_engineering.py      # Engineered features
│   │   ├── loader.py                   # Data loading
│   │   ├── orchestrator.py             # Main class managing all steps
│   │   ├── scaling.py                  # Scaling (StandardScaler)
│   │   └── target_builder.py           # Formation of target variables
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── trainer.py
│   │
│   ├── src/utils/
│   │   ├── __init__.py                 # Exports all functions for backward compatibility
│   │   ├── config.py                   # load_config, save_config
│   │   ├── file.py                     # save_dataframe, load_dataframe, save_pickle, load_pickle
│   │   ├── logging.py                  # setup_logging
│   │   └── seed.py                     # set_seed
│   │
│   ├── __init__.py
│   ├── infer.py
│   └── model.py
│
├── tests/
│   ├── test_data_generation.py
│   ├── test_infer.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_train.py
│
├──visualization/
│   └── plots/
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract base class
│   │   ├── damage.py                   # Damage plot
│   │   ├── probability.py              # T1 plot
│   │   ├── risk_factors.py             # Risk factors
│   │   └── scatter.py                  # Risk map
│   │
│   ├── __init__.py
│   ├── dashboard.py                    # Orchestrator
│   ├── data_provider.py                # Singleton for loading predictions
│   └── theme.py                        # Colors, fonts, style
│
├── .gitignore
├── LICENSE
├── main.py
├── README.en.md
├── README.md
└── requirements.txt
```

---

## 🚀 Quick start

### 📋 Prerequisites
- Python 3.10 or newer
- It is recommended to use a virtual environment

### ⚙️ Installation and setup

```bash
git clone https://github.com/MordaRedFox/ai-insurance-cyber-risk-forecast.git
cd ai-insurance-cyber-risk-forecast
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### 🖥️ Running the console application

```bash
python main.py
```

### 🔧 Running individual scripts

```bash
python run_scripts/run_generation.py
python run_scripts/run_preprocessing.py
python run_scripts/run_training.py
python run_scripts/run_inference.py
python run_scripts/run_visualize.py
```

### 🧪 Running tests

```bash
pytest tests/ -v
```

---

## 📩 Contacts
I am open to constructive criticism and suggestions for improving the code. If you found an error or know how to make something better - please contact me!

[![Telegram](https://img.shields.io/badge/-MordaRedFox-0088cc?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/MordaRedFox)
&nbsp;
[![Email](https://img.shields.io/badge/-mordaredfox@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mordaredfox@gmail.com)
