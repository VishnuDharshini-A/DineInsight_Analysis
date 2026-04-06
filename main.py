import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

#DATAEXPLORATION AND PREPROCESSING

df=pd.read_csv("restaurant.csv")
print(df.head())
print(df.info())
print(df.shape)

#checking missing values
print(df.isnull().sum())

#analysing distribution of target variables

print(df['Aggregate rating'].value_counts())
print(df["Aggregate rating"].value_counts(normalize=True)*100)

df["Aggregate rating"].value_counts().plot(kind="bar")
plt.title("Rating distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()
#
df["Rating Category"]=df["Aggregate rating"].apply(lambda x:"Not Rated" if x==0 else "Rated")
print(df["Rating Category"].value_counts())

df["Rating Category"].value_counts().plot(kind="bar")
plt.xticks(rotation=0)
plt.title("Rating Category Distribution")
plt.show()

#DESCRIPTIVE ANALYSIS

print(df.describe())

#distribution of country code

print(df['Country Code'].value_counts())
df['Country Code'].value_counts().plot(kind='bar')
plt.title('Country Code Distribution')
plt.xlabel('Country code')
plt.ylabel('count')
plt.show()

#distribution of cities

print(df['City'].value_counts())
df['City'].value_counts().head(10).plot(kind='bar')
plt.title("Top cities distribution")
plt.xlabel('City')
plt.ylabel('count')
plt.show()

#distribution of cuisines

df['Cuisines']=df['Cuisines'].str.split(',')
df_exploded=df.explode('Cuisines')
print(df_exploded['Cuisines'].value_counts().head(10))

df_exploded['Cuisines'].value_counts().head(10).plot(kind='bar')
plt.title('Top cuisines')
plt.xlabel('country')
plt.ylabel('counts')
plt.show()

#GEOSPATIAL ANALYSIS

#Distribution of restaurants

plt.figure(figsize=(12,8))
m=Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='grey',lake_color='white')
m.drawmapboundary(fill_color='lightblue')
x,y=m(df['Longitude'].values,df['Latitude'].values)
sc=m.scatter(x,y,c=df['Aggregate rating'],cmap='viridis',alpha=0.5)
plt.colorbar(sc,label='Rating')
plt.title("Global Distribution of restaurants with ratings")
plt.show()

#Distribution of restaurants in countries

df['Country Code'].value_counts().plot(kind='bar')
plt.title("Restaurant Distribution by country")
plt.show()

#correlation between location and rating

corr=df[['Latitude','Longitude','Aggregate rating']].corr()
print(corr)
sns.heatmap(corr,annot=True)
plt.title("Location vs Rating Correlation")
plt.show()

#CUSTOMER PREFERENCE ANALYSIS

#relationship between type of cuisine and restaurant rating
df['Cuisines']=df['Cuisines'].str.split(',')
df_exploded=df.explode('Cuisines')
print(df_exploded)

df_exploded.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
df_exploded.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Most popular cuisines based on average rating")
plt.xticks(rotation=45)
plt.show()

#popular cuisines based on votes
df_exploded.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(10)
df_exploded.groupby('Cuisines')['Votes'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Most popular cuisines based on votes")
plt.xticks(rotation=45)
plt.show()

#usual high rating cuisines
avg_rating=df_exploded.groupby('Cuisines')['Aggregate rating'].mean()
avg_rating=avg_rating.sort_values(ascending=False)
avg_rating.head(10)

avg_rating.head(10).plot(kind='bar')
plt.title("Top cuisines with highest rating")
plt.xticks(rotation=45)
plt.show()

#PREDICTIVE MODELING

df=pd.get_dummies(df,drop_first=True)

target='Aggregate rating'
x=df.drop(target,axis=1)
y=df['Aggregate rating']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)

Scaler=StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_test=Scaler.transform(x_test)

models={
    'Linear Regression':LinearRegression(),
    'Decision Tree':DecisionTreeRegressor(),
    'Random Forest':RandomForestRegressor(random_state=42,n_jobs=-1)
}

results={}

for name,model in models.items():
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)

    mae=mean_absolute_error(y_test,predictions)
    mse=mean_squared_error(y_test,predictions)
    r2=r2_score(y_test,predictions)

    results[name]=r2

    print(name)
    print('mae:',mae)
    print('mse:',mse)
    print('r2:',r2)

best_model=max(results,key=results.get)
print('best model:',best_model)
print('best r2:',results[best_model])


#DATA VISUALIZATION

#rating distribution using multiple charts

#histogram
plt.hist(df['Aggregate rating'],bins=20)
plt.title("Distribution of restaurant ratings")
plt.xlabel("rating")
plt.ylabel("count")
plt.show()

#bar plot
df['Aggregate rating'].value_counts().sort_index().plot(kind='bar')
plt.title("count of each rating")
plt.xlabel("rating")
plt.ylabel("No of restaurants")
plt.show()

#box plot
sns.boxplot(y='Aggregate rating',data=df)
plt.title("Rating distribution")
plt.show()

#kde plot
sns.kdeplot(df['Aggregate rating'],fill=True)
plt.title("Rating Density Distribution")
plt.show()

#various features vs target

#votes vs rating
sns.scatterplot(x='Votes',y='Aggregate rating',data=df)
plt.title("Votes vs Rating")
plt.show()

#cost vs rating
sns.scatterplot(x='Average Cost for two',y='Aggregate rating',data=df)
plt.title("Cost vs Rating")
plt.show()

#price range vs rating
sns.boxplot(x='Price range',y='Aggregate rating',data=df)
plt.title("Price range vs Rating")
plt.show()

#online delivery vs rating
sns.boxplot(x='Has Online delivery',y='Aggregate rating',data=df)
plt.title("Online delivery vs Rating")
plt.show()

#table booking vs rating
sns.boxplot(x='Has Table booking',y='Aggregate rating',data=df)
plt.title("Table booking vs Rating")
plt.show()

#city vs rating
df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("city vs rating")
plt.show()

#cuisine vs Rating
df_exploded.groupby("Cuisines")['Aggregate rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("cuisine vs rating")
plt.show()

#correlation heatmap
corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True)
plt.title("Correlation with ratings")
plt.show()


