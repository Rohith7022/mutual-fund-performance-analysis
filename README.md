# Mutual Fund Performance Analysis
## VCU – Advanced Financial Analytics (FIRE 691)

Comprehensive performance evaluation of actively managed mutual funds using **CRSP Mutual Fund Database (2005–2025)** and **Fama-French 5-Factor + Momentum Model** with rolling window estimation.

---

## Overview

Analyzed fund performance persistence, factor exposures, and manager skill using a large-scale panel dataset. Rolling 3-year window estimates reveal whether alpha is persistent or driven by luck.

---

## Data Sources

- CRSP Mutual Fund Database (monthly returns, TNA, expense ratio, turnover)
- Ken French Data Library (FF5 + Momentum factors)

---

## Metrics

- 4-factor and 5-factor alpha
- Sharpe Ratio and Appraisal Ratio
- R-squared and factor loading stability
- Rolling 36-month window estimates
- Index fund vs. active fund comparison

---

## Key Findings

- Most actively managed funds underperform passive benchmarks on a risk-adjusted basis after expenses
- Fund size (TNA) and turnover ratio are significant predictors of future performance
- Rolling alphas reveal that outperformance is rarely persistent across full sample periods

---

## Tech Stack

`Python` `pandas` `numpy` `statsmodels` `matplotlib` `WRDS/CRSP`

---
*Virginia Commonwealth University - MS Business (Financial Analytics) - FIRE 691*
