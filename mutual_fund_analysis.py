# ============================================================
# Mutual Fund Performance Analysis
# VCU - Advanced Financial Analytics (FIRE 691) - Week 8
# Author: Rohith Ravindra Reddy
# Data: CRSP Mutual Fund Database 2005-2025, FF5+MOM model
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')
from google.colab import drive
drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/FINANCIAL ANALYTICS Colab Notebooks/'

# ── LOAD CRSP MUTUAL FUND DATA ───────────────────────────────
funds = pd.read_csv(PATH+'mutual_fund_returns.csv')
funds.columns = funds.columns.str.lower()
funds['date'] = pd.to_datetime(funds['date'])
funds['yyyymm'] = funds['date'].dt.year*100 + funds['date'].dt.month
print(f"Fund-months: {len(funds):,}")
print("Columns:", funds.columns.tolist())

# ── LOAD FF5 + MOMENTUM FACTORS ──────────────────────────────
ff5 = pd.read_csv(PATH+'ff5_factors.csv', skiprows=3)
ff5.columns = ['yyyymm','Mkt-RF','SMB','HML','RMW','CMA','RF']
ff5 = ff5[ff5['yyyymm'].astype(str).str.len()==6].copy()
ff5['yyyymm'] = ff5['yyyymm'].astype(int)
for c in ff5.columns[1:]: ff5[c] = pd.to_numeric(ff5[c],errors='coerce')/100

mom = pd.read_csv(PATH+'momentum_factor.csv')
mom.columns = ['yyyymm','MOM']
mom['yyyymm'] = mom['yyyymm'].astype(int)
mom['MOM'] = pd.to_numeric(mom['MOM'],errors='coerce')/100
factors = ff5.merge(mom,on='yyyymm',how='inner')

# ── MERGE FUNDS WITH FACTORS ──────────────────────────────────
merged = funds.merge(factors,on='yyyymm',how='inner')
merged['excess_ret'] = merged['ret'] - merged['RF']

# ── COMPUTE FULL-SAMPLE METRICS PER FUND ─────────────────────
factor_cols = ['Mkt-RF','SMB','HML','RMW','CMA','MOM']

def fund_metrics(g):
    if len(g) < 36: return None
    X = sm.add_constant(g[factor_cols])
    try:
        m = sm.OLS(g['excess_ret'], X).fit()
        alpha_ann = m.params['const']*12
        sharpe = (g['excess_ret'].mean()/g['excess_ret'].std())*np.sqrt(12)
        resid_std = m.resid.std()*np.sqrt(12)
        return pd.Series({'alpha_ann':alpha_ann,'sharpe':sharpe,
                          'appraisal':alpha_ann/resid_std if resid_std>0 else np.nan,
                          'r2':m.rsquared,'n_months':len(g)})
    except: return None

fund_id = 'crsp_fundno' if 'crsp_fundno' in merged.columns else 'fundno'
metrics = (merged.groupby(fund_id).apply(fund_metrics).dropna())
print(f"\nFunds analysed: {len(metrics):,}")
print("\n=== Performance Metrics Distribution ===")
print(metrics[['alpha_ann','sharpe','r2']].describe().round(4))

pct_pos_alpha = (metrics['alpha_ann']>0).mean()*100
print(f"\n% Funds with positive alpha: {pct_pos_alpha:.1f}%")
print("Key finding: Most actively managed funds underperform after expenses.")

# ── ALPHA DISTRIBUTION PLOT ───────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(16,5))
for i,(col,title) in enumerate([('alpha_ann','FF5+MOM Alpha (Annualised)'),('sharpe','Sharpe Ratio'),('r2','R-squared')]):
    axes[i].hist(metrics[col].clip(-1,1),bins=60,color='navy',edgecolor='white',alpha=0.8)
    axes[i].axvline(0,color='red',lw=1.5,ls='--')
    axes[i].set_title(title); axes[i].set_xlabel(col); axes[i].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(PATH+'fund_metrics_dist.png',dpi=150); plt.show()

# ── ROLLING 36-MONTH ALPHA (AGGREGATE) ───────────────────────
# Average alpha across all funds per month
roll_alpha = []
months = sorted(merged['yyyymm'].unique())
for i in range(36,len(months)):
    window_months = months[i-36:i]
    w = merged[merged['yyyymm'].isin(window_months)]
    if len(w)<100: continue
    try:
        X = sm.add_constant(w[factor_cols])
        m = sm.OLS(w['excess_ret'],X).fit()
        roll_alpha.append({'yyyymm':months[i],'alpha':m.params['const']*12})
    except: pass

ra_df = pd.DataFrame(roll_alpha)
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(ra_df['yyyymm'],ra_df['alpha']*100,color='navy',lw=1.5)
ax.axhline(0,color='red',ls='--',lw=1); ax.set_title('Rolling 36-Month Aggregate Fund Alpha (%)')
ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(PATH+'rolling_fund_alpha.png',dpi=150); plt.show()
print("\nKey finding: Rolling alphas show alpha is rarely persistent across full sample.")
