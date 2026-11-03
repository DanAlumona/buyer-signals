"""Ready buyer or casual browser? Which session signals predict a purchase,
and how much predictive power is lost when a signal goes missing.

Data: UCI Online Shoppers Purchasing Intention (Sakar et al., 2018), CC BY 4.0.
12,330 e-commerce sessions; Revenue = the session ended in a purchase.

Run:  .venv/Scripts/python src/analysis.py
Writes reports/results.json and reports/*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
SEED = 7

NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATEGORICAL = ["Month", "VisitorType", "TrafficType", "Weekend"]

# Signals grouped the way an advertiser would recognise them.
SIGNAL_GROUPS = {
    "value signal (PageValues)": ["PageValues"],
    "engagement depth (product pages + time)": ["ProductRelated", "ProductRelated_Duration"],
    "drop-off signals (bounce + exit rates)": ["BounceRates", "ExitRates"],
    "visitor context (new vs returning, source, month)": ["VisitorType", "TrafficType", "Month"],
}


def load():
    df = pd.read_csv(ROOT / "data" / "online_shoppers_intention.csv")
    df["bought"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(str)
    df["TrafficType"] = df["TrafficType"].astype(str)
    return df


def compare_buyers_and_browsers(df):
    """RQ1: how far apart are buyers and browsers on each numeric signal?
    Welch t-test for the difference in means, Mann-Whitney because the data
    is heavily skewed, and Cohen's d so effects can be compared."""
    rows = []
    buyers, browsers = df[df.bought == 1], df[df.bought == 0]
    for col in NUMERIC:
        a, b = buyers[col], browsers[col]
        t = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"signal": col, "buyer_mean": round(a.mean(), 3),
                     "browser_mean": round(b.mean(), 3),
                     "cohens_d": round((a.mean() - b.mean()) / pooled, 2),
                     "welch_p": float(t.pvalue), "mann_whitney_p": float(mw.pvalue)})
    return sorted(rows, key=lambda r: -abs(r["cohens_d"]))


def conversion_by_segment(df):
    """Conversion rate by categorical segment, with a chi-square test of
    independence for each."""
    out = {}
    for col in ["VisitorType", "Weekend", "Month"]:
        table = pd.crosstab(df[col], df.bought)
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = (table[1] / table.sum(axis=1)).round(3).sort_values(ascending=False)
        out[col] = {"conversion_rate": rates.to_dict(), "chi2": round(chi2, 1),
                    "dof": int(dof), "p": float(p)}
    return out


def anova_page_value(df):
    """One-way ANOVA: does the value signal differ by visitor type?"""
    groups = [g.PageValues.values for _, g in df.groupby("VisitorType")]
    f = stats.f_oneway(*groups)
    return {"F": round(f.statistic, 1), "p": float(f.pvalue)}


def logistic_odds_ratios(df):
    """RQ2: holding the other signals constant, how much does each one move
    the odds of purchase? Inputs are standardised so odds ratios are per
    1 standard deviation."""
    d = df.copy()
    cols = ["PageValues", "ProductRelated_Duration", "ExitRates", "BounceRates",
            "Administrative", "Informational"]
    for c in cols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    model = smf.logit("bought ~ " + " + ".join(cols) + " + C(VisitorType) + C(Weekend)",
                      data=d).fit(disp=False)
    ci = model.conf_int()
    rows = [{"term": term, "odds_ratio": round(float(np.exp(model.params[term])), 2),
             "ci_low": round(float(np.exp(ci.loc[term, 0])), 2),
             "ci_high": round(float(np.exp(ci.loc[term, 1])), 2),
             "p": float(model.pvalues[term])}
            for term in model.params.index if term != "Intercept"]
    return {"pseudo_r2": round(float(model.prsquared), 3), "terms": rows}


def make_model(num, cat):
    pre = make_column_transformer(
        (StandardScaler(), num),
        (OneHotEncoder(handle_unknown="ignore"), cat))
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def fit_auc(train, test, num, cat):
    m = make_model(num, cat).fit(train[num + cat], train.bought)
    return roc_auc_score(test.bought, m.predict_proba(test[num + cat])[:, 1])


def signal_loss(df):
    """RQ3: what is each signal group worth? Drop one group at a time and
    measure the fall in AUC on held-out sessions (ablation)."""
    train, test = train_test_split(df, test_size=0.3, stratify=df.bought, random_state=SEED)
    full = fit_auc(train, test, NUMERIC, CATEGORICAL)
    out = {"full_auc": round(full, 3), "drop_group": {}}
    for name, cols in SIGNAL_GROUPS.items():
        num = [c for c in NUMERIC if c not in cols]
        cat = [c for c in CATEGORICAL if c not in cols]
        out["drop_group"][name] = round(full - fit_auc(train, test, num, cat), 3)
    return out


def coverage_curve(df):
    """RQ3b: signals rarely vanish completely; coverage degrades. Blank the
    value signal for a growing share of sessions (as if the tag or server
    event failed to fire) and track AUC. The model has no way to know which
    sessions lost the signal, as in real ingestion."""
    train, test = train_test_split(df, test_size=0.3, stratify=df.bought, random_state=SEED)
    rng = np.random.default_rng(SEED)
    curve = []
    for share in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        tr, te = train.copy(), test.copy()
        for frame in (tr, te):
            lost = rng.random(len(frame)) < share
            frame.loc[lost, "PageValues"] = 0
        curve.append({"share_missing": share,
                      "auc": round(fit_auc(tr, te, NUMERIC, CATEGORICAL), 3)})
    return curve


def plot(results):
    d = results["buyers_vs_browsers"][:6]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([r["signal"] for r in d][::-1], [r["cohens_d"] for r in d][::-1], color="#4c72b0")
    ax.set_xlabel("Effect size, buyers vs browsers (Cohen's d)")
    ax.set_title("Which signals separate buyers from browsers")
    fig.tight_layout(); fig.savefig(OUT / "effect_sizes.png", dpi=150); plt.close(fig)

    c = results["coverage_curve"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([p["share_missing"] * 100 for p in c], [p["auc"] for p in c], marker="o", color="#c44e52")
    ax.set_xlabel("Sessions missing the value signal (%)")
    ax.set_ylabel("AUC on held-out sessions")
    ax.set_title("Predictive power as signal coverage degrades")
    fig.tight_layout(); fig.savefig(OUT / "coverage_curve.png", dpi=150); plt.close(fig)


def main():
    df = load()
    results = {
        "sessions": len(df),
        "purchase_rate": round(df.bought.mean(), 3),
        "buyers_vs_browsers": compare_buyers_and_browsers(df),
        "segments": conversion_by_segment(df),
        "anova_pagevalue_by_visitor": anova_page_value(df),
        "logistic": logistic_odds_ratios(df),
        "signal_loss": signal_loss(df),
        "coverage_curve": coverage_curve(df),
    }
    plot(results)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ["sessions", "purchase_rate", "signal_loss", "coverage_curve"]}, indent=1))


if __name__ == "__main__":
    main()
