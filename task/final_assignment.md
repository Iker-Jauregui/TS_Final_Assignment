# Advanced Time Series Forecasting · Final Assignment

## Spain Energy Demand Forecasting Challenge

*A National-Scale Forecasting Problem for the Spanish Electricity System*

| | |
|---|---|
| **Dataset** | Hourly Energy Demand, Generation & Weather in Spain (Kaggle) |
| **Period** | 2015 – 2018 · Hourly resolution |
| **Target** | National electricity demand (MWh per hour) |
| **Core Challenge** | Beat the official TSO day-ahead forecast |

Thousands of engineers at grid operators across Europe work every day to forecast electricity demand as accurately as possible — because every megawatt-hour of error has a direct financial and environmental cost. This capstone project is not a classroom exercise with a predetermined answer. It is an open research challenge. There is no single correct model. Your task is to understand the problem deeply, explore the data honestly, make deliberate technical choices, and defend them with evidence.

---

## 1. Business Context & Your Role

Your work is to develop a forecasting system capable of predicting hourly national electricity demand across the Spanish peninsula.

The TSO already publishes its own official day-ahead demand forecast every day. This forecast, produced by a team of experienced engineers using proprietary models and decades of operational knowledge, is the professional baseline you have been asked to improve. Your models will be evaluated not only against the actual historical demand, but explicitly against the performance of this official forecast. Outperforming it — even partially, even on specific time windows or demand regimes — is the definition of success in this engagement.

**Why does this matter?** In a modern electricity system, an error of even 1,000 MWh in a single hour can trigger emergency balancing actions and force purchases on the intraday spot market at elevated prices. In the most severe cases, a sustained forecasting failure can cascade into a system-wide frequency imbalance — pushing the grid beyond its operational limits and precipitating a total blackout. The complete collapse of a national electricity system, however rare, has severe consequences and may seriously affect critical infrastructure and cause economic disruption that can take days to recover from. This is not a theoretical exercise: it is the exact problem that organisations like REE, Elia, RTE, and National Grid face every single day.

---

## 2. The Dataset

The dataset for this project is publicly available on Kaggle under the title:

**"Hourly Energy Demand, Generation and Weather in Spain"**

Available at: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

The dataset covers the period 2015–2018 at hourly resolution and contains two linked files that you will need to merge and explore carefully:

| File | Key Variables | Notes |
|------|--------------|-------|
| `energy_dataset.csv` | - `total load actual` → your prediction target<br>- `total load forecast` → the official TSO forecast to beat<br>- Generation mix by technology (wind, solar, nuclear, hydro…)<br>- Cross-border energy flows<br>- Day-ahead electricity market price (EUR/MWh) | All columns are at hourly frequency. Some contain missing values and anomalies that must be handled explicitly. Forecasted values (load, generation, market price) **cannot** be used as prediction covariates. |
| `weather_features.csv` | - Temperature, humidity, wind speed, wind direction<br>- Weather description (categorical)<br>- Pressure, cloud cover<br>- Data for 5 major Spanish cities | One row per city per hour. Must be aggregated or selected appropriately before use as covariates. |

---

## 3. Problem Formulation

Your objective is to build a forecasting system that generates accurate, reliable predictions of hourly national electricity demand for Spain. The precise scope of the problem — the forecast horizon, the prediction granularity, the covariates included — is part of your research process. However, the following constraints apply to all submissions:

1. Your system must predict `total load actual` at hourly resolution.
2. Your final model must be evaluated on a clearly defined hold-out test set that was never used during model training, hyperparameter tuning, or feature selection.
3. Your submission must include a rigorous quantitative comparison between: (a) your model's predictions, (b) the actual historical demand, and (c) the official TSO forecast included in the dataset. All three must be reported on the same test period using the same metrics.
4. You must use at least two different error metrics in your evaluation. Your choice of metrics must be justified in relation to the business context.
5. The use of prediction intervals in your forecast is strongly encouraged. If so, you should additionally explore possible metrics to measure the precision of the given prediction intervals.

---

## 4. Research Freedom & Covariates

This project has no prescribed solution path. You are expected to apply the full breadth of knowledge and methodologies covered in this course, and even go beyond them, making deliberate, evidence-based decisions about what to include and what to discard.

The choice of forecasting approach is entirely yours. You are expected to explore, prototype, compare and select. There is no single right answer — but there is a wrong process: choosing a model without understanding why, and reporting results without understanding what they mean.

### Going Beyond the Dataset (External Covariates)

One of the most professionally valuable skills in applied forecasting is knowing when the data you have is not enough, and being able to source and integrate additional information that improves predictive power. You are strongly encouraged to enrich your feature set with external covariates beyond those already present in the dataset. However, if you consider new covariates in your model, you must verify that they would genuinely be available at prediction time in a real operational context.

---

## 5. Evaluation Criteria

Your submission will be evaluated across five dimensions. Note that technical accuracy is necessary but not sufficient: the quality of your reasoning, the rigour of your methodology, and the clarity of your communication matter.

- **Exploratory Data Analysis & Problem Understanding:** Quality and depth of the initial data exploration. Identification of data quality issues. Clarity of the problem framing and the choices it motivates.

- **Modelling Strategy & Critical Justification:** Rigour and originality of the approach selected. You must explain why you chose your final architecture — not just what it is. What alternatives did you consider and discard, and on what evidence? How does your approach address the specific challenges identified? Models adopted without critical reasoning will be penalised.

- **Validation Methodology & Benchmark Comparison:** Correctness and transparency of the train/validation/test protocol. Explicit evidence that no data leakage occurred. Clear quantitative comparison between your model, the actual demand, and the TSO official forecast on the same test period. Discussion of on which time windows or conditions your model outperforms or underperforms the TSO benchmark.

- **Feature Engineering & Use of External Covariates:** Quality and creativity of the feature engineering process. If external covariates were incorporated, are they properly motivated, sourced, and validated? Teams that meaningfully enrich the feature set with well-reasoned external information will be recognised.

- **Results Interpretation & Business Communication:** Ability to translate technical results into actionable insights. What does your model's performance mean for the TSO? Where does it fail, and why? What would you need to improve it further?

---

## 6. Deliverables

Each group must submit the following documents:

| # | Deliverable | Description | Format |
|---|------------|-------------|--------|
| D1 | **Technical Notebook** | A fully self-contained, reproducible Jupyter notebook that covers the complete pipeline: data ingestion and cleaning, exploratory analysis, feature engineering, model training, validation, and evaluation. The notebook must run end-to-end without errors on a clean Python environment. All random seeds must be set. External data sources must be documented and retrievable. | `.ipynb` |
| D2 | **Technical Report** | A structured written report documenting your methodology, choices, results, and conclusions. Must include an executive summary, and sections covering the corresponding evaluation criteria exposed above. | `.pdf` (max. 12 pp.) |

---

## 7. Academic Integrity & Use of AI Tools

You are permitted and encouraged to use AI coding assistants (e.g., GitHub Copilot, ChatGPT, Claude) to support your work. The use of these tools is considered standard professional practice. However:

- You are **fully responsible** for understanding every line of code in your submitted notebook. During the Q&A session, instructors will ask you to explain and modify specific parts of your implementation.
- **AI-generated text may not be submitted as-is** in your written report. Your analysis, interpretation, and conclusions must reflect your own reasoning. Sections that read as generic AI output rather than domain-specific analysis will be penalised.
- **Results must not be fabricated or cherry-picked.** If your best model underperforms relative to the TSO baseline, you should report it honestly and analyse the reasons why. A technically honest negative result, well-argued, earns more credit than inflated numbers with no critical reflection.
