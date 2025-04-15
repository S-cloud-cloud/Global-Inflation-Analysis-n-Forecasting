result = adfuller(Global_avg_inflation)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
