import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import os
#Global Inflation Analysis (1970 - 2023)

cleaned_file = r'C:\Users\KIIT\Global_Inflation_Analysis\cleaned_proj_data.csv'
#update_file_path here 

#checking if cleaned data already exists or is needed:
if os.path.exists(cleaned_file):
    #check regeneration required:
    Gen_csv = input(" Regenerate A CSV file of cleaned data ? : yes/no : \t ").strip().lower()
    if Gen_csv == "yes" : 
        re_gen_csv = True
    else:
        re_gen_csv = False
else:
    re_gen_csv = True #file does not exist : user has only raw file

if re_gen_csv:
    
    print('running data cleaning process : \n')
    file_path = r'C:\Users\KIIT\Global_Inflation_Analysis\Data_xls\Inflation-data.xlsx'
    #raw_data_workbook file path to be updated

    #contains multiple sheets, some of which has metadata
    workbook = pd.ExcelFile(file_path,engine="openpyxl")
    #check if the first row is header:
    #df = pd.read_excel(file_path, engine='openpyxl', header=1)  # Adjust index if needed
    print('\n workbook.sheet_names : \n', workbook.sheet_names)  

    # List all sheet names

    #['Intro', 'top', 'hcpi_m', 'hcpi_q', 'hcpi_a', 'ecpi_m', 'ecpi_q', 'ecpi_a', 'fcpi_m',
    # 'fcpi_q', 'fcpi_a', 'ccpi_m', 'ccpi_q', 'ccpi_a', 'ppi_m', 'ppi_q', 'ppi_a', 'def_q', 'def_a', 'Aggregate']

    # Read the desired sheet: [0-based indexing]
    #1.For overall inflation trends across countries → Use hcpi_a (Annual Harmonized CPI).
    #2.For inflation vs. oil price analysis → Use ecpi_a (Annual Energy CPI).
    #3.For a high-level summary before deeper analysis → Use Aggregate.

    #1.
    proj_data=pd.read_excel(file_path,sheet_name='hcpi_a',engine="openpyxl")
    #data_exploration
    print('\n sheet choosen : hcpi_a : first 6 lines of hcpi_a: \n', proj_data.head(6),'\n') 

    print('\n Data Type info : \n', proj_data.info(),'\n')
    #Number of rows/columns : 59 (0-based/60)
    #Data types (numeric, text, etc.) : dtypes: float64(55), object(5) 
    #Missing values 

    #print('\n All columns',proj_data.columns,'\n') #data of years from 1970 to 2023

    print('checking column series name : \n',proj_data['Series Name'].head(6),'\n') #Headline Consumer Price Inflation : includes both energy and food prices

    #Data Cleaning & Preprocessing:
    print('checking missing values per col count : \v', proj_data.isnull().sum(),'\n')
    #multiple missing values

    print('show all NaN values \n', proj_data[proj_data.isnull().any(axis=1)], '\n')

    # Counts empty string values in each column
    print(f"empty string values in each column : \n,{(proj_data==' ').sum()},\n")


    print(f"Count of duplicate records : \v {proj_data.duplicated().sum()} \n")
    #zero duplicates : else : proj_data = proj_data.drop_duplicates()

    #col wise null values : percentage : 
    print('col wise null values : percentage : \n', proj_data.isnull().mean() * 100) 
    #none of the cols have more than 50% null values

    #checking for patterens in missing data:
    print("Checking if any pattern in missing data exists : \n")
    plt.figure(figsize=(10,6))
    sns.heatmap(proj_data.isnull(), cmap='Blues', cbar=True, yticklabels=False)
    plt.show()

    #for just country column:
    print('\n Checking country wise missing data pattern \n')
    plt.figure(figsize=(10,6))
    sns.heatmap(proj_data[['Country']].isnull(), cmap='viridis', cbar=True, yticklabels=False)
    plt.show()
    #less effective in such huge data with low missing values
    print(f"\n missing country wise data : \n {proj_data['Country'].isnull()} \n")

    print('Threshold= 50`%` of total row count :\v ', len(proj_data)*0.5)
    print(f"\n count of non-null values per col : \v {proj_data.count()}\n") 
    #so no col will be dropped by using : proj_data.dropna(thres = len(proj_data)* 0.5,axis=1) : 
    # as all have more than 50% non-null values

    #since numerical data col is time series data : forward-fill missing values :
    #should not do ffill on entire data : categorical cols exist : proj_data = proj_data.fillna(method='ffill')

    #since categorical cols have less than 1 % missing values and there exists diverse data as country & country code col
    #replace missing categorical data with 'unknown' keeps data neutral

    #pick dtypes from info 
    #seperate different dtypes of col
    categorical_col = proj_data.select_dtypes(include=['object']).columns
    numerical_col = proj_data.select_dtypes(include=['float64']).columns

    proj_data[categorical_col] = proj_data[categorical_col].fillna("unknown")

    #proj_data[numerical_col]=proj_data[numerical_col].apply(pd.to_numeric, errors='coerce')
    
    #forward and backward fill :
    proj_data[numerical_col]=proj_data[numerical_col].ffill() 
    #in case missing value is present in row1:
    proj_data[numerical_col]=proj_data[numerical_col].bfill()

    #cleaned up data:
    print('\n count of values in each col after clean-up :\n', proj_data.isnull().count()) #205 for all cols
    print(f"\n null count : \n {proj_data.isnull().sum().sum()}\n")#1
    print(f"\n null count : \n {proj_data.isnull().sum()}\n") #1 in 2023
    #optional : here data is small enough for a mannual check:
    print('\nchecking values in year 2023 : \n', proj_data[[2023]].to_string())

    #in order to ensure no hidden null are there:
    #Sometimes, columns may have "hidden" NaN values if they’re stored as objects (strings) instead of numeric types.
    proj_data[numerical_col]=proj_data[numerical_col].apply(pd.to_numeric, errors='coerce')
    print(f"\n null count : \n {proj_data.isnull().sum()}\n") #0 in 2023
    print('\nchecking values in year 2023 : \n', proj_data[[2023]].to_string())
    print(f"\n null count : \n {proj_data.isnull().sum().sum()}\n")#0
    #it wasn’t a truly missing value but a misformatted numeric value that got properly converted.
    #pd.to_numeric(), it fixed the formatting issue and converted it into a proper number.
    #takeaway : apply pd.to_numeric before Filling in NaN values.

    print(proj_data.info())
    filename_u = input("What do you wanna name your file ?\t")
    proj_data.to_csv(f"{filename_u}.csv", index=False)
    print("\n Cleaned data saved as CSV")

else:
    proj_data = pd.read_csv(cleaned_file)
    print("\nCleaned data loaded from CSV, Success!")

    
#EDA phase 
eda_phase = input(" Do an Exploratory data analysis ? : yes/no : \t ").strip().lower()
if eda_phase =='yes' :
    print(" Project in EDA Phase : \n")
    print(proj_data.info(),'\n\n')
    print(proj_data.head(3),'\n\n')
    #1. Mean, Median, Standard Deviation 

    #1.
    print("Statistics : \n")
    print(proj_data.describe())#.to_string())
    print(f"\n checking the no of unique countries : \t { proj_data['Country'].nunique() } \n") #204 : all unique countries
    #optional : data small enooough to see
    print(proj_data[['Country']].to_string(),'\n')
    print(proj_data.columns)


    #Time series data analysis :
    #Goal : analysing inflation over the decades, convert data to fit all the years in a single col
    #i.e. change data from wide format to long format : such that col : Country , Year and Inflation are Included
    #melting : wide to long: 

    year_col = [str(year) for year in range(1970, 2024)] #2024 not included
    col_in_use = ["Country"] + year_col #created a list 

    analysis_col = proj_data[col_in_use]
    projdata_long = analysis_col.melt(id_vars=["Country"], var_name="Year", value_name="hcpia_Inflation")
    #to ensure proper numerical formatting
    projdata_long['Year'] = projdata_long['Year'].astype(int)
    print('\n', projdata_long.head(3))
    print('\n', projdata_long.tail(10))
    print(f"\n {len(projdata_long)}")
    print('\n\n', projdata_long.info())

    print(f"\n \n Data description : \n {projdata_long.describe()} \n")
    print(f"\n \n Inflation data contained is for years:  {projdata_long['Year'].min()} to {projdata_long['Year'].max()} \n")

    #2.
    print("\n Inflation Trends Over Time \n ")
    #print(projdata_long.sort_values(by='hcpia_inflation', ascending=False).head(10))

    Global_avg_inflation = projdata_long.groupby("Year")["hcpia_Inflation"].mean()
    #since we are plotting time series data next , reset_index() is not used after groupby 
    # so we can avoid declaring the x index while plotting as 'Year' is the default index here now
    print("Global Average Inflation per year : \n",Global_avg_inflation)

    print("\n PLOTTING Global Inflation Trend: \n")

    plt.figure(figsize=(10,6))
    #x : global_avg_inflation.index : (year); y : Gloabal_avg_inflation.values : (avg_inflation_rate)
    print("\n Plot info : x : global_avg_inflation.index : (year); y : Gloabal_avg_inflation.values : (avg_inflation_rate) \n")
    plt.plot(Global_avg_inflation,marker="o",linestyle="-",color="b", label="Global avg inflation rate vs year")
    plt.xlabel("Year")
    plt.ylabel("Average Inflation Rate")
    plt.title("Global Inflation Trend over time \n")
    plt.legend()
    plt.grid(True)
    plt.show()

    #3.
    print("\n Finding the countries with highest and lowest inflation in 2023 : \n")
    latest_year = projdata_long['Year'].max()

    highest_inflation = projdata_long[projdata_long['Year']== latest_year].nlargest(10, "hcpia_Inflation")
    lowest_inflation = projdata_long[projdata_long['Year']==latest_year].nsmallest(10, "hcpia_Inflation")

    #highest inflation rate
    print(f"\n Top 10 countries with the highest inflation rate in : {latest_year} \n ")
    plt.figure(figsize=(12 , 5 ))
    sns.barplot(y=highest_inflation['Country'], x=highest_inflation['hcpia_Inflation'], palette='Reds_r',
                hue=highest_inflation['Country'],legend=False)
    plt.xlabel("Inflation Rate")
    plt.ylabel("Country")
    plt.title(f"Countries with the highest inflation rate in {latest_year}")
    plt.show()


    #lowest inflation rate
    print(f"\n 10 countries with the lowest inflation rate in : {latest_year} \n ")
    plt.figure(figsize=(12 , 5 ))
    sns.barplot(y=lowest_inflation['Country'], x=lowest_inflation['hcpia_Inflation'],hue=lowest_inflation['Country'], palette='Blues_r',
                legend = False)
    plt.xlabel("Inflation Rate %")
    plt.ylabel("Country")
    plt.title(f"Countries with the lowest inflation rate in {latest_year}")
    plt.show()

    #aiming for a dynamic, year-wise comparison of the top inflating countries to see trends over time. 
    print("\n Highest and lowest inflation countries per year : \n")
    #for small data : alt way : top_countries = 
    # projdata_long.groupby('Year').apply(lambda x: x.nlargest(10, 'hcpia_Inflation')).reset_index(drop=True)

    #faster for large data : sorting , group_by : head(10)
    #Sort by year and Inflation descending 
    #top_countries = > projdata_long.groupby('Year') -> nlargest(5, 'hcpia_Inflation')
    #bottom_countries = projdata_long.groupby("Year") -> nsmallest(5, 'hcpia_Inflation')

    #using sort_values and then slicing for faster processing with a bigger dataset here
    highest_per_year = ( projdata_long.sort_values(['Year','hcpia_Inflation'],ascending=[True,False]) ).groupby('Year').head(1) 
    #df now contains top 5 inflaction rate countries for each year and that year as index

    lowest_per_year = (projdata_long.sort_values(['Year', 'hcpia_Inflation'],ascending=[True, True])).groupby('Year').head(1)
    #df now contains coutries with the lowest inflation for every year, using year as index    

    #creating a custom color palette for each country: 
    # unq_country = highest_per_year["Country"].nunique()
    # palette_c = sns.color_palette("hls",unq_country) 

    print("Plotting Countries with the highest inflation rate per year : \n")
    plt.figure(figsize=(14 , 8))
    sns.lineplot(data=highest_per_year,x='Year',y='hcpia_Inflation',hue='Country', marker='o')
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate %")
    plt.title("Top Inflating countries over time")
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Puts legend outside
    # plt.tight_layout()
    plt.grid(True)
    plt.show()

    print("\n Plotting countries with the lowest inflation rate per year : \n")
    plt.figure(figsize=(14,8))
    sns.lineplot(data=lowest_per_year,x='Year',y='hcpia_Inflation',hue='Country',marker='o',linestyle='-')
    plt.xlabel('Year')
    plt.ylabel("Inflation rate %")
    plt.title("Leat Inflating countries over time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout ()
    #plt.legend()
    plt.grid(True)
    plt.show()

    #4.
    print("Plotting World Inflation Heatmap: \n ")

    #creating a choropleth map for inflation rates
    fig = px.choropleth ( projdata_long[projdata_long['Year']==latest_year], locations='Country', locationmode = 'country names', color = 'hcpia_Inflation', hover_name = 'Country',title = f'{latest_year} Global Inflation Heatmap', color_continuous_scale = px.colors.sequential.Plasma )

    fig.show()
    
    #5.
    print("Inflation Distribution Analysis : ")
    #Histogram to see inflation distribution
    print("Plotting Inflation distribution rate using a histogram  \n")
    
    plt.figure(figsize=(12,7))
    sns.histplot(projdata_long['hcpia_Inflation'],bins=30,kde=True,color='purple')
    plt.xlabel("Inflation Rate %")
    plt.ylabel("Frequency")
    plt.title("Inflation Rate distribution")
    plt.show()
    
    print("Plotting a boxplot to identify the outliers \n")
    
    plt.figure(figsize=(12,7))
    sns.boxplot(x=projdata_long['hcpia_Inflation'],color='orange')
    plt.title('Boxplot of Inflation Rates (Detecting outliers)')
    plt.show()
    
else : 
    print("EDA phase deactivated \n ")


print('Time-series forecasting and predicting future inflation \n')
        
#1.
print('Time series visualization : (rolling averages and Trends) :visual smoothing \n')

plt.figure(figsize=(12,6))

#plot the original graph year vs avg_inflation
plt.plot(Global_avg_inflation,label='Original', color='blue')

#ploting avg_inlation for every 5 years , to see a smoother data with less noise and random spikes 
# in order to understand the trend of inflation
plt.plot(Global_avg_inflation.rolling(window=5).mean(),label="5-year rolling avg", color='red',linewidth='3')

plt.title('Global inflation trend - smoothed ')
plt.xlabel('Year')
plt.ylabel('Inflation rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#2.
#auto_regressive model 
#Auto Regressive: Uses past values to predict future ones
#Integrated: Makes the time series stationary by differencing it
#MA: Uses past forecast errors to improve predictions

#ARIMA model
#ARIMA helps us forecast future values in a time series by learning patterns from the past values and their errors.
print("Using gloabal_avg_inflation : a clean, non-volatile trend.")

#check if data is stationary as ARIMA works best on stationary data : constant mean & variance over time
print("using the  Augmented Dickey-Fuller (ADF) test to check data is stationary or not ?: ")

from statsmodels.tsa.stattools import adfuller
print(Global_avg_inflation.head())
result = adfuller(Global_avg_inflation)
print(f"ADF Statistic: {result[0]}") #more negative : more stationary  #-4.37 : less than threshold of 1%
pvalue = result[1]
print(f"p-value: {result[1]}") #
if(pvalue < 0.05):
    setARIMA = True
    print('stationary data : use ARIMA as-is : 1,1,1')
else:
    print('Data is not stationary')

if setARIMA:
    from statsmodels.tsa.arima.model import ARIMA
    model= ARIMA(Global_avg_inflation, order=(1,1,1))
    model_fit = model.fit()
    print(model_fit.summary()) #future values are dependent on past values \
        
print("Model summary shows future values dependent on past values ")
print('Forecasting future 10 years using the ARIMA model :\n')
forecast = model_fit.forecast(steps=10)

#xlabel : creating an index of 10 years in future 
forecasted_for_years = np.arange(Global_avg_inflation.index.max()+1,Global_avg_inflation.index.max() + 11)
#creates a range of 10 years

print("Plotting a comparison b/w historical inflation data and current predicted future Inflation : \n")
plt.figure(figsize=(16,6))
#plt.plot(Global_avg_inflation,label='Historical Inflation',color="blue",linewidth='2')

#plotting the forecasted future inflation     

plt.plot(forecasted_for_years,forecast,label=f'Forecast for next ten years : {latest_year+1} to {latest_year+10}', color='red',linestyle='--',marker='o')
plt.title("Global Inflation Forecast (AIRMA Model)")
plt.xlabel("Year")
plt.ylabel("Predicted Inflation Rate")
plt.legend()
plt.grid(True)

# plt.xticks(rotation=45)
# plt.xticks(ticks=np.arange(1970,2023,5))

plt.tight_layout()
plt.show()

plt.figure(figsize=(18,6))
plt.plot(Global_avg_inflation.index, Global_avg_inflation.values,label='Historical Inflation : 1970-2023',color="blue",linewidth='2')
plt.plot(forecasted_for_years,forecast,label=f'Forecast for next ten years : {latest_year+1} to {latest_year+10}', color='red',linestyle='--',marker='o',linewidth=2,markersize=6)

plt.scatter(forecasted_for_years,forecast,color='green',zorder=5)
plt.xlabel("Year")
plt.ylabel("Predicted Inflation Rate : hcpia")
plt.title("Global Inflation Forecast (AIRMA Model)")
plt.xticks(np.arange(1970,2023,5),rotation=45)
plt.grid(True,linestyle='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


print("\n Feature engineering for Inflation Analysis \n ")

#1.
#convert global trend series to a dataframe
#converting to data frame so that we can add columns like lag_values, rolling averages
Avg_inflation_df = Global_avg_inflation.reset_index()
Avg_inflation_df.columns = ['Year' , 'Avg_hcpia_Inflation']

#2.
#lag values : this is a time series , past values strongly affect the future values 
#lag values : Giving model memory of inflation_value of just a year ago to help with better prediction

#memory of past year's inflation : for every year its avg_inflation in past year is being stored
Avg_inflation_df['Inflation_Lag1'] = Avg_inflation_df['Avg_hcpia_Inflation'].shift(1)
#memory of past year's past year's inflation : for every year its avg_inflation for 2 years before is being stored
Avg_inflation_df['Inflation_Lag2'] = Avg_inflation_df['Avg_hcpia_Inflation'].shift(2)

print('Lag features used to capture memory of past inflation to model/predict future inflation better \n')

#3.
print('To smooth out short-term noise and highlight long-term trends : adding rolling features/ moving averages to reduce variance and outliers in the raw inflation data \n')
#avoids sudden spikes or drops , adds a trendiness : better for model prediction
Avg_inflation_df["Inflation_MA3"] = Avg_inflation_df['Avg_hcpia_Inflation'].rolling(window=3).mean()
Avg_inflation_df['Inflation_MA5'] = Avg_inflation_df['Avg_hcpia_Inflation'].rolling(window=5).mean()

#scope to plot a visual comparision graph here

#4.
print('Precentage Change in Inflation (decline/growth) year over year \n')
#Better trend direction , volatility detection to check for shocks in the data , to check feature usefulness
Avg_inflation_df['Inflation_pct_change']=Avg_inflation_df['Avg_hcpia_Inflation'].pct_change() * 100

plt.figure(figsize=(12,5))
plt.plot(Avg_inflation_df['Year'],Avg_inflation_df['Inflation_pct_change'],color='red', marker='o',linestyle='--')
plt.axhline(y=0, color='grey', linestyle='--') #a reference line  at 0
plt.title('Year-over-Year percentage change in Inflation')
plt.xlabel('Year')
plt.ylabel('Percentage change in Inflation')
plt.grid(True)
plt.tight_layout()
plt.show()

#Above 0 : Comparitively increased , below 0 : comparitevly decresed , big spikes or drops : shocks

print(Avg_inflation_df.tail(10))


#5.
print('Interactive summary visualization phase : \n')

#1.
print('Interactive Line plot : Global Inflation Trend \n')

fig = px.line(Avg_inflation_df, x ='Year', y = 'Avg_hcpia_Inflation', title='Interactive Global Inflation Trend')
fig.update_traces(line=dict(color='blue'))
fig.update_layout(yaxis_title='Inflation Rate%', xaxis_title='Year')
fig.show()

#2.
print('Heatmap of Inflation : Country VS year \n')

#countries as rows , years as cols : pivot into heatmap_data
# projdata_long = analysis_col.melt(id_vars=["Country"], var_name="Year", value_name="hcpia_Inflation")
# Aggregate duplicate entries
projdata_grouped = projdata_long.groupby(["Country", "Year"])["hcpia_Inflation"].mean().nlargest(20).reset_index()

heatmap_data = projdata_grouped.pivot(index='Country',columns='Year',values='hcpia_Inflation')

#plotting the heatmap
fig = px.imshow(heatmap_data, aspect='auto', color_continuous_scale='RdYlGn_r', title='Heatmap of Inflation rates by Country (1970-2023)',labels=dict(x='Year',y='Country',color='Inflation rate %'))

fig.update_layout(
    xaxis_title='Year',yaxis_title='Country',autosize=False,width=1000,height=1400,
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()

#5.3 animated graphs : future scope








