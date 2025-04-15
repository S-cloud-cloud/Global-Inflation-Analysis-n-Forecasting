<h1 align="center"> Global Inflation Analysis (1970–2023) & Forecasting </h1>

<p align=center>
This project explores worldwide inflation from 1970 to 2023, performs in-depth exploratory analysis, and builds a forecasting model using ARIMA to predict global inflation for the next 10 years.<br>

A deep dive into historical inflation patterns across 200+ countries using data analytics and time-series forecasting. <br>
Powered by Python | Forecasted with ARIMA | Covers 50+ Years | prediction for a decade
 </p>

------

## Table of contents 
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Goals](#project-goals)
- [EDA Highlights](#eda-highlights)
- [Forecasting Approach](#forecasting-approach)
- [Visual Samples](#visual-samples)
- [Key Insights](#key-insights)
- [Challenges Faced](#challenges)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)
- [Data and Envm setup](#Data-and-Envm-setup)
- [How to Run](#how-to-run)

---
## <a name="Overview"></a>  Overview

<h2> Overview :
 This project analyzes historic global inflation data (1970–2023) for over 200 countries, reveals patterns through exploratory data analysis (EDA), and utilizes the ARIMA model for forecasting the next decade of inflation trends. It is built for analysts, students, and economic researchers interested in macroeconomic behavior and predictive analytics.
 </h2>

---
## <a name="tech-stack"></a> Tech Stack 

| Category       | Tools/Libraries 
|----------------|-----------------
| Language       | Python          
| Libraries      | `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `sklearn` 
| Forecasting    | ARIMA           
| IDE            | VS Code    
| Format         | Jupyter Notebook (`.ipynb`) 

---

## <a name="project-goals"></a> Project Goals

- Analyze global hcpia inflation trends using the world bank data across 200+ countries over 5 decades.
- Use EDA to uncover hidden trends, anomalies, and relationships, trend visualization.
- Forecast future inflation using time-series modeling.
- Feature engineering for smoother visualizations.
- Communicate results through clean, interactive visualizations.

---
## <a name="eda-highlights"></a> EDA Highlights

- Inflation trends by continent and top 10 economies.
- Global average inflation.
- yearly change over time.
- Finding the countries with highest and lowest avg inflation in 2023.
- year-wise comparison of the top and least inflating countries to see trends over time.
- Heatmaps for country-wise inflation by decade (World Inflation Heatmap)
- Outlier detection for hyperinflation events.
- Volatility patterns and event correlations.

---
## <a name="forecasting-approach"></a> Forecasting Approach

- **Model Used**: ARIMA (1,1,1)
- **Target Variable**: `Global_avg_inflation`
- Converted `Year` to `datetime` and set frequency to `'YS'` (year-start).
- Forecasted 10 years into the future with confidence intervals.

---
## <a name="visual-samples"></a> Visual Samples
 <h3 align="center"> A few visual samples from the analysis <h3>

<p align="center">
  <img src="\images\Global_inflation_trend_over_time.png" width="600"/><br>
  <em>Global Inflation Trend Over Years</em>
</p>

<p align="center">
  <img src="images\Global_Inflation_heatmap.png" width="600"/><br>
  <em>Avg hcpia Global Inflation heatmap 2023</em>
</p>

<p align="center">
  <img src="images\Countries_with_highest_inflation_rate_in_2023.png" width="600"/><br>
  <em>Countries_with_highest_inflation_rate_in_2023</em>
</p>

<p align="center">
  <img src="images\2023_countries_with_lowest_inflation_rate.png" width="600"/><br>
  <em>countries_with_lowest_inflation_rate_in_2023</em>
</p>

<p align="center">
  <img src="images\Global_inflation vs rolling avg smoothed global inflation.png" width="600"/><br>
  <em>Global_inflation vs rolling avg smoothed global inflation</em>
</p>

<p align="center">
  <img src="images\Global_Inflation_forecast.png" width="600"/><br>
  <em>Global_Inflation_forecast</em>
</p>

<p align="center">
  <img src="images\YOY pct change.png" width="600"/><br>
  <em>Year over Year percentage change in avg Inflation rate</em>
</p>

---

## <a name="key-insights"></a> Key Insights <sub><em>Below are the key insights from the analysis : </em></sub>

1.Countries like Venezula, Zimbabwe,Argentina and Sudan faced extreme inflation crises in multiple decades.
2.Trends of Global avg hcpia_Inflation : 
   <h3 align='center'> Big Rise Around 2017–2018 </h3>
    <p>This could reflect inflation surges in multiple countries (like emerging economies or oil-driven economies).Possible global triggers: Commodity price spikes, trade tensions (e.g., US-China trade war beginnings), or regional instability.</p>

   <h3 align='center'> Drop in 2019–2020 </h3>
       <p>Likely influenced by:
       Global slowdown in 2019 (manufacturing decline, trade tensions).
       COVID-19 in 2020 → economic shutdowns, demand destruction → low inflation or even deflation in some areas.
       </p>
    
  <h3 align ='center'> Flat Around 2021 Onwards </h3>
      <p> Governments began stimulus packages → demand revived → inflation returned.
      You’ll probably see inflation rise again post-2021 — a result of supply-chain crunch, energy price shocks, and war in Ukraine.
      </p>
3.Inflation remained moderate and stable in the early 2000s.
4.Inflation spiked post-2020, likely due to global disruptions (e.g., COVID-19).
5.The ARIMA model suggests a moderate Global aveg inflation trend in the coming decade, assuming historical patterns persist.

---

<a name="challenges-faced--your-input-needed"></a>Challenges Faced 

<ul>
  <li><strong>Data Preprocessing Complexity:</strong> Standardizing inflation data from 200+ countries over 50+ years required careful reshaping (wide to long format) and handling of missing/unknown values.</li>
  <li><strong>Time-Series Indexing Issues:</strong> During the ARIMA modeling, transitioning from <code>.py</code> to <code>.ipynb</code> format introduced unexpected errors due to datetime indexing and frequency recognition (<code>YS</code> vs <code>YS-JAN</code>).</li>
  <li><strong>Cross-Platform Consistency:</strong> Certain visualizations and forecasting code behaved differently across environments (VS Code vs Jupyter Notebook), especially regarding forecast steps and frequency assumptions.</li>
  <li><strong>Maintaining Clean Visual Aesthetics:</strong> Balancing clarity and aesthetics for charts (e.g., heatmaps, line plots, confidence intervals) while making them insightful was a major design consideration.</li>
  <li><strong>ARIMA Parameter Selection:</strong> Choosing the right (p,d,q) order required several iterations and visual diagnostics to get a model that reasonably forecasted future inflation trends.</li>
</ul>

---

<a name="conclusion"></a> Conclusion 
<p>
  This Global Inflation Analysis project served as a full-spectrum application of data science — from raw data transformation to deep statistical modeling and forecasting. By analyzing inflation trends across decades and nations, we uncovered key patterns, historical anomalies, and modeled plausible future scenarios using ARIMA. 
</p>

<p>
  The project not only strengthened foundational skills in EDA, time-series, and visualization but also instilled an appreciation for macroeconomic data storytelling. With the structured forecasting model in place, this work serves as a solid base for evolving into real-time dashboards, economic scenario planning tools, and more advanced modeling in future iterations.
</p>

---

<a name="future-scope"></a>Future Scope
Integrate with Streamlit/Dash for real-time dashboards.

Connect live IMF/World Bank APIs for updated forecasts.

Compare ARIMA with models like Prophet, XGBoost, LSTM.

Add correlation layers for GDP, interest rate, and unemployment.

---
<a name="Data-and-Envm-setup"></a>Data and Envm setup

<h2 id="data-setup">📁 Data & Environment Setup</h2>

<p><strong>📚 Research Paper & Dataset Source:</strong><br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0261560623000979" target="_blank">
ScienceDirect – Harmonized Inflation Dataset (1970–2023)
</a></p>

<hr>

<h3>Understanding the Sheets in the Dataset Workbook</h3>

<p>Refer to <strong>Sheet 2</strong> of the Excel file for a dictionary of all other sheets. Here's a breakdown:</p>

<ul>
  <li><strong>hcpi_m / hcpi_q / hcpi_a</strong> → Harmonized Consumer Price Index (monthly / quarterly / annual)</li>
  <li><strong>ecpi_*:</strong> Energy CPI (focuses on fuel/electricity inflation)</li>
  <li><strong>fcpi_*:</strong> Food CPI</li>
  <li><strong>ccpi_*:</strong> Core CPI (excludes food and energy)</li>
  <li><strong>ppi_*:</strong> Producer Price Index (inflation at production level)</li>
  <li><strong>def_q / def_a:</strong> GDP Deflator (broad measure)</li>
  <li><strong>Aggregate:</strong> Summary of all indices</li>
</ul>

<h4>Recommended Sheets:</h4>
<ul>
  <li><code>hcpi_a</code> for overall annual inflation analysis</li>
  <li><code>ecpi_a</code> for inflation vs energy comparison</li>
  <li><code>Aggregate</code> for high-level overview</li>
</ul>


<h3>How to Load the Dataset in Python</h3>

file_path = "your_file.xlsx"  # Replace with your actual path
inflation_data = pd.read_excel(file_path, sheet_name="hcpi_a", engine="openpyxl")
print(inflation_data.head())

To load a different sheet, simply change "hcpi_a" to the relevant sheet name.

<h3>Environment Setup</h3> <p><strong>Required Python Libraries:run in terminal</strong></p>
 <p> pip install pandas numpy matplotlib seaborn plotly statsmodels openpyxl</p>
<hr>
<h3 align='center'>Library	Purpose :</h3>

pandas	                | Data manipulation ;
numpy	                | Numerical computing ;
matplotlib & seaborn	| Static visualizations ;
plotly	                | Interactive graphs ;
statsmodels	            | Statistical modeling (ARIMA) ;
openpyxl	            | Excel support for .xlsx files ;

<hr>
<h3>Setting Up VS Code for Analytics</h3> 
<h3> For .ipynb (Jupyter) Support in VS Code </h3>
  <p>pip install notebook jupyter,Then create a .ipynb notebook</p>
  <li>If graphs don't show in `.py` script, use <code>plt.show()</code>.</li> </ul>

---
<a name="how-to-run"></a>How to Run

git clone https://github.com/S-cloud-cloud/Global-Inflation-Analysis-n-Forecasting.git
<br>
cd Global-Inflation-Analysis-n-Forecasting
<br>
jupyter notebook

---
