import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt
def Clear_file():
    Rtl_data = pd.read_excel("/home/rabit/PycharmProjects/pythonProject/OnlineRetail.xlsx")
    Rtl_data.head()
    country_cust_data=Rtl_data[['Country','CustomerID']].drop_duplicates()
    country_cust_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
    #Keep only United Kingdom data
    Rtl_data = Rtl_data.query("Country=='United Kingdom'").reset_index(drop=True)
    #Check for missing values in the dataset
    print(Rtl_data.isnull().sum(axis=0))
    #Remove missing values from CustomerID column, can ignore missing values in description column
    Rtl_data = Rtl_data[pd.notnull(Rtl_data['CustomerID'])]
    #Validate if there are any negative values in Quantity column
    print(Rtl_data.Quantity.min())
    #Validate if there are any negative values in UnitPrice column
    print(Rtl_data.UnitPrice.min())
    #Filter out records with negative values
    Rtl_data = Rtl_data[(Rtl_data['Quantity']>0)]
    #Convert the string date field to datetime
    Rtl_data['InvoiceDate'] = pd.to_datetime(Rtl_data['InvoiceDate'])
    #Add new column depicting total amount
    Rtl_data['TotalAmount'] = Rtl_data['Quantity'] * Rtl_data['UnitPrice']
    #Check the shape (number of columns and rows) in the dataset after data is cleaned
    print(Rtl_data.shape)
    print(Rtl_data.head())
    return Rtl_data
def RFM_Futre():
    #Recency = Latest Date - Last Inovice Data, Frequency = count of invoice no. of transaction(s), Monetary = Sum of Total
    #Amount for each customer
    Rtl_data=Clear_file()
    import datetime as dt

    #Set Latest date 2011-12-10 as last invoice date was 2011-12-09. This is to calculate the number of days from recent purchase
    Latest_Date = dt.datetime(2011,12,10)
    #Create RFM Modelling scores for each customer
    RFMScores = Rtl_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})

    #Convert Invoice Date into type int
    RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)

    #Rename column names to Recency, Frequency and Monetary
    RFMScores.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalAmount': 'Monetary'}, inplace=True)

    print(RFMScores.reset_index().head())
    out = RFMScores
    out.to_csv("customerSegmentation.csv")
    return RFMScores
import seaborn as sns
def Figure():
    RFMScores= pd.read_csv("/home/rabit/PycharmProjects/pythonProject/customerSegmentation.csv")
    #Descriptive Statistics (Recency)
    RFMScores.Recency.describe()
    plt.figure(figsize=(9, 9))
    #Recency distribution plot
    plt.subplot(3, 1, 1)
    x = RFMScores['Recency']
    ax = sns.distplot(x)
    #Descriptive Statistics (Frequency)
    RFMScores.Frequency.describe()
    plt.subplot(3, 1, 2)
    x1= RFMScores.query('Frequency < 1000')['Frequency']
    ax1 = sns.distplot(x1)
    #Descriptive Statistics (Monetary)
    RFMScores.Monetary.describe()
    plt.subplot(3, 1, 3)
    x2 = RFMScores.query('Monetary < 10000')['Monetary']
    ax2 = sns.distplot(x1)
    plt.show()
#Figure()
#Functions to create R, F and M segments
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1
def RFM_Segmentation():
    RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/customerSegmentation.csv")
    quantiles = RFMScores.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()
    #Calculate Add R, F and M segment value columns in the existing dataset to show R, F and M segment values
    RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
    RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
    RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
    #Calculate and Add RFMGroup value column showing combined concatenated score of RFM
    RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)
    #Calculate and Add RFMScore value column showing total sum of RFMGroup values
    RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
    print(RFMScores.head())
    #Assign Loyalty Level to each customer
    Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
    Score_cuts = pd.qcut(RFMScores.RFMScore, q = 4, labels = Loyalty_Level)
    RFMScores['RFM_Loyalty_Level'] = Score_cuts.values
    out= RFMScores
    out.to_csv("segmentrfm.csv")
def Log_dfm(num):
    if num<=0:
        return 1
    else:
        return num
def Figure_RFM_Log():
    RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/segmentrfm.csv")
    #Apply handle_neg_n_zero function to Recency and Monetary columns
    RFMScores['Recency'] = [Log_dfm(x) for x in RFMScores.Recency]
    RFMScores['Monetary'] = [Log_dfm(x) for x in RFMScores.Monetary]

    #Perform Log transformation to bring data into normal or near normal distribution
    Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)
    #Data distribution after data normalization for Recency
    plt.figure(figsize=(9, 9))
    plt.subplot(3, 1, 1)
    Recency_Plot = Log_Tfd_Data['Recency']
    ax = sns.distplot(Recency_Plot)
    #Data distribution after data normalization for Frequency
    plt.subplot(3, 1, 2)
    Frequency_Plot = Log_Tfd_Data.query('Frequency < 3000')['Frequency']
    ax = sns.distplot(Frequency_Plot)
    #Data distribution after data normalization for Monetary
    plt.subplot(3, 1, 3)
    Monetary_Plot = Log_Tfd_Data.query('Monetary < 30000')['Monetary']
    ax = sns.distplot(Monetary_Plot)
    plt.show()
def Log_da_Tfd_Data():
    RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/segmentrfm.csv")
    #Apply handle_neg_n_zero function to Recency and Monetary columns
    RFMScores['Recency'] = [Log_dfm(x) for x in RFMScores.Recency]
    RFMScores['Monetary'] = [Log_dfm(x) for x in RFMScores.Monetary]

    #Perform Log transformation to bring data into normal or near normal distribution
    Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)
    out=Log_Tfd_Data
    out.to_csv("Log_rfm.csv")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
def Figure_data_after():
    RFMScores =pd.read_csv("/home/rabit/PycharmProjects/pythonProject/customerSegmentation.csv")
    RFMScores_fr=RFMScores[['Recency','Frequency']]
    ax=RFMScores_fr.plot(
       kind="scatter",
       x="Recency", y="Frequency",
       figsize=(10,8),c='b');
    plt.show()

def Figure_Elow():
   Log_Tfd_Data=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Log_rfm.csv")
   RFMScores= pd.read_csv("/home/rabit/PycharmProjects/pythonProject/customerSegmentation.csv")
   #Bring the data on same scale
   scaleobj = StandardScaler()
   Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

   #Transform it back to dataframe
   Scaled_Data = pd.DataFrame(Scaled_Data, index =RFMScores.index, columns = Log_Tfd_Data.columns)
   #print(Scaled_Data.head())
   sum_of_sq_dist = {}
   for k in range(1,15):
       km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
       km = km.fit(Scaled_Data)
       sum_of_sq_dist[k] = km.inertia_

   #Plot the graph for the sum of square distance values and Number of Clusters
   sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
   plt.xlabel('Number of Clusters(k)')
   plt.ylabel('Sum of Square Distances')
   plt.title('Elbow Method For Optimal k')
   plt.show()
def Customer_rfm_Kmean():
   Log_Tfd_Data=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Log_rfm.csv")
   RFMScores= pd.read_csv("/home/rabit/PycharmProjects/pythonProject/customerSegmentation.csv")
   #Bring the data on same scale
   scaleobj = StandardScaler()
   Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)
   #Transform it back to dataframe
   Scaled_Data = pd.DataFrame(Scaled_Data, index =RFMScores.index, columns = Log_Tfd_Data.columns)
   #Perform K-Mean Clustering or build the K-Means clustering model
   KMean_clust = KMeans(n_clusters= 4, init= 'k-means++', max_iter= 1000)
   KMean_clust.fit(Scaled_Data)

       #Find the clusters for the observation given in the dataset
   RFMScores['Cluster'] = KMean_clust.labels_
   out=RFMScores
   out.to_csv("Custormer_rfm_Kmean.csv")
   #print(RFMScores.head())
from matplotlib import pyplot as plt
def Figure_KMeanCluser():
    #plt.figure(figsize=(7,7))
    RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Custormer_rfm_Kmean.csv")
    #Scatter Plot Frequency Vs Recency
    Colors = ["red", "green", "blue","yellow"]
    RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
    ax = RFMScores.plot(
       kind="scatter",
       x="Recency", y="Frequency",
       figsize=(10,8),
       c = RFMScores['Color'])
    plt.show()
def Display():
    RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Custormer_rfm_Kmean.csv")
    Colors = ["red", "green", "blue","yellow"]
    RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
    Out=RFMScores
    Out.to_csv("Clusering.csv")

"""
RFMScores=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Clusering.csv")
#print(RFMScores.head())
plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
dataPie=RFMScores.groupby('Cluster')['Frequency'].sum()
dataPie.plot.pie(autopct="%.1f%%");
plt.subplot(3, 1, 2)
dataPie=RFMScores.groupby('Cluster')['Recency'].sum()
dataPie.plot.pie(autopct="%.1f%%");
plt.subplot(3, 1, 3)
dataPie=RFMScores.groupby('Cluster')['Monetary'].sum()
dataPie.plot.pie(autopct="%.1f%%")
plt.show()
"""

if __name__ == '__main__':
    Figure_data_after()
    Figure_KMeanCluser()
