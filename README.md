# buyer-signals

Which signals separate a ready buyer from a casual browser, what each signal is worth, and how much predictive power is lost as signal coverage degrades. The readout is in [reports/FINDINGS.md](reports/FINDINGS.md).

## Run it
```bash
python -m venv .venv
.venv/Scripts/python -m pip install pandas scipy statsmodels scikit-learn matplotlib
curl -L -o data/uci468.zip "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
(cd data && unzip uci468.zip)
.venv/Scripts/python src/analysis.py
```

Data: Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2018), *Real-time prediction of online shoppers' purchasing intention*, UCI Machine Learning Repository, CC BY 4.0.
