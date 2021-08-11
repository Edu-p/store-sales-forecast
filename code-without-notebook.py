# 0.0 Imports

import math
import numpy  as np
import pandas as pd
import random
import pickle
import warnings
# import inflection
import seaborn as sns
import xgboost as xgb

from scipy                 import stats  as ss
from boruta                import BorutaPy
from matplotlib            import pyplot as plt
from IPython.display       import Image
from IPython.core.display  import HTML


from sklearn.metrics       import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, accuracy_score
from sklearn.ensemble      import RandomForestRegressor
from sklearn.linear_model  import LinearRegression, Lasso
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
import datetime
import requests

warnings.filterwarnings( 'ignore' )

## 0.1. Helper Functions

def cross_validation( x_training, kfold, model_name, model, verbose=False ): # verbose serve para se queremos que printe cada interaçao da funçao do kfold 
    mae_list = []
    mape_list = []
    rmse_list = []
    for k in reversed( range( 1, kfold+1 ) ):
        if verbose:
            print( '\nKFold Number: {}'.format( k ) )
        # start and end date for validation 
        validation_start_date = x_training['date'].max() - datetime.timedelta( days=k*6*7)
        validation_end_date = x_training['date'].max() - datetime.timedelta( days=(k-1)*6*7)

        # filtering dataset
        training = x_training[x_training['date'] < validation_start_date]
        validation = x_training[(x_training['date'] >= validation_start_date) & (x_training['date'] <= validation_end_date)]

        # training and validation dataset
        # training
        xtraining = training.drop( ['date', 'sales'], axis=1 ) 
        ytraining = training['sales']

        # validation
        xvalidation = validation.drop( ['date', 'sales'], axis=1 )
        yvalidation = validation['sales']

        # model
        m = model.fit( xtraining, ytraining )

        # prediction
        yhat = m.predict( xvalidation )

        # performance
        m_result = ml_error( model_name, np.expm1( yvalidation ), np.expm1( yhat ) )

        # store performance of each kfold iteration
        mae_list.append(  m_result['MAE'] )
        mape_list.append( m_result['MAPE'] )
        rmse_list.append( m_result['RMSE'] )

    return pd.DataFrame( {'Model Name': model_name,
                          'MAE CV': np.round( np.mean( mae_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mae_list ), 2 ).astype( str ),
                          'MAPE CV': np.round( np.mean( mape_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mape_list ), 2 ).astype( str ),
                          'RMSE CV': np.round( np.mean( rmse_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( rmse_list ), 2 ).astype( str ) }, index=[0] ) # np.round faz mesma coisa do .2f em c e aqui em python

def mean_percentage_error( y, yhat ):
    return np.mean( ( y - yhat ) / y )

def cramer_v( x, y ):
    cm = pd.crosstab( x, y )
    cm = cm.values
    n = cm.sum()
    r, k = cm.shape
    
    chi2 = ss.chi2_contingency( cm )[0]
    chi2corr = max( 0, chi2 - (k-1)*(r-1)/(n-1) )
    
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)
    
    return np.sqrt( (chi2corr/n) / ( min( kcorr-1, rcorr-1 ) ) )

def ml_error( model_name, y, yhat ):
    mae = mean_absolute_error( y, yhat )
    mape = mean_absolute_percentage_error( y, yhat )
    rmse = np.sqrt( mean_squared_error( y, yhat ) )
    
    return pd.DataFrame( { 'Model Name' : model_name,
                           'MAE': mae,
                           'MAPE': mape,
                           'RMSE': rmse }, index=[0] ) # rmse mede o erro 

## 0.2.Loading data

df_sales_raw = pd.read_csv( 'C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/datasets/train.csv', low_memory=False)
df_store_raw = pd.read_csv( 'C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/datasets/store.csv', low_memory=False)

# merge
df_raw = pd.merge( df_sales_raw, df_store_raw, how='left', on='Store' )

# 1.0 Data description


def jupyter_settings():    
    sns.set()
jupyter_settings()

df1 = df_raw.copy()

## 1.1. Rename Columns

cols_new = [ 'store','day_of_week','date','sales','customers','open','promo','state_holiday','school_holiday','store_type','assortment','competition_distance','competition_open_since_month','competition_open_since_year','promo2','promo2_since_week','promo2_since_year','promo_interval' ]

# renaming
df1.columns = cols_new

df1.columns

## 1.2. Data Dimensions

print( 'number of Rows: {}'.format( df1.shape[0] ) )
print( 'number of Rows: {}'.format( df1.shape[1] ) )


## 1.3. Data Types

df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes

## 1.4. Check NA

df1.isna().sum()

## 1.5. Fillout NA

# competition_distance
df1['competition_distance'] = df1['competition_distance'].apply( lambda x: (df1['competition_distance'].max())*10 if math.isnan( x ) else x )


# competition_open_since_month
df1['competition_open_since_month'] = df1.apply ( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )


# competition_open_since_year
df1['competition_open_since_year'] = df1.apply ( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )


# promo2_since_week
df1['promo2_since_week'] = df1.apply ( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )



# promo2_since_year
df1['promo2_since_year'] = df1.apply ( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )

# promo_interval
month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

df1['promo_interval'].fillna(0, inplace=True )

df1['month_map'] = df1['date'].dt.month.map( month_map )

df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )


df1.isna().sum()

df1['competition_distance'].max()

df1.sample(5).T # transposto(inverte linha e coluna)

## 1.6 Change Types

print( df1.dtypes )


df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )


## 1.7 Descriptive Statistical

num_attributes = df1.select_dtypes( include=['int64', 'float64'] )
cat_attributes = df1.select_dtypes( exclude=['int64', 'float64', 'datetime64[ns]'] )


### 1.7.1 Numerical Attributes

# Central Tendency -  mean, median
ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T

# Dispersion - std, min, max, range, skev, kurtosis
d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T
d2 = pd.DataFrame( num_attributes.apply( min ) ).T 
d3 = pd.DataFrame( num_attributes.apply( max ) ).T 
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T 
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T 
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis ) ).T 

# concatenate
dfDesc = pd.concat( [ d2, d3, d4, ct1, ct2, d1, d5, d4 ] ).T.reset_index()

dfDesc.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']

print(dfDesc)

sns.distplot( df1['competition_distance'] )

### 1.7.2 Categorical Attributes

aux1 = df1[ ( df1['state_holiday'] != '0' ) & ( df1['sales'] > 0 )] 

plt.subplot( 1,3,1 )
sns.boxplot( x='state_holiday' , y='sales' , data=aux1 )

plt.subplot( 1,3,3 )
sns.boxplot( x='store_type' , y='sales' , data=aux1 )


# 2.0 Feature Engineering

df2 = df1.copy()
Image( 'img/MindMapHypothesis.png' )

## 2.1 Mind map Hypothesis


### 2.1.1 Store Assumptions


** 1. ** Stores with higher number of seller employees sell more. (I don't have number of employees in this dataset)

** 2. ** Stores with higher stock capacity sell more. (I don't have stock info)

** 3. ** Larger sized stores sell more. 

** 4. ** Stores with larger company assortments sell more.

** 5. ** Stores with more practice competitors sell less.

** 6. ** Stores with competitors for the longest school years sell more.




### 2.1.2 Product Assumptions

**1.** Stores that invest more in Marketing should sell more. (I have no marketing expenses (budget))

**2.** Stores with greater product exposure should sell more. (I don't have a quantity of products with exposure)

**3.** Stores with products with a lower price should sell more. (I don't have product prices)

**5.** Stores with more aggressive promotions (bigger discounts), should sell more. (We don't have promotion categories)

**6.** Stores with promotions active for longer should sell more.

**7.** Stores with more promotion days should sell more.

**8.** Stores with more consecutive promotions should sell more.


### 2.1.3 Time Assumptions


**1.** Lojas abertas durante o feriado de Natal deveriam vender mais.

**2.** Lojas deveriam vender mais ao longo dos anos.

**3.** Lojas deveriam vender mais no segundo semestre do ano.

**4.** Lojas deveriam vender mais depois do dia 10 de cada mês.

**5.** Lojas deveriam vender menos aos finais de semana.

**6.** Lojas deveriam vender menos durante os feriados escolares.


## 2.2 Final Assumptions

#### Final Store List

**1.** Stores with larger assortments should sell more.

**2.** Stores with closer competitors should sell less.

**3.** Stores with longer-term competitors should sell more.


#### Final product List

**4.** Stores with promotions active for longer should sell more.

**5.** Stores with more promotion days should sell more.

**7.** Stores with more consecutive promotions should sell more.


#### Final time List

**8.** Stores open during the Christmas holiday should sell more.

**9.** Stores should sell more over the years.

**10.** Stores should sell more in the second half of the year.

**11.** Stores should sell more after the 10th of each month.

**12.** Stores should sell less on weekends.

**13.** Stores should sell less during school holidays.

## 2.3 Feature Engineering


# year
df2['year'] = df2['date'].dt.year

# month
df2['month'] = df2['date'].dt.month

# day
df2['day'] = df2['date'].dt.day

# week of year
df2['week_of_year'] = df2['date'].dt.isocalendar().week

# year week
df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

# # competition since
df2['competition_open_since_month'] = df2['competition_open_since_month'].astype( int )
df2['competition_open_since_year'] = df2['competition_open_since_year'].astype( int )
df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

# assortment
df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

# state holiday
df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )


# promo since
df2['promo2_since_week'] = df2['promo2_since_week'].astype( int )
df2['promo2_since_year'] = df2['promo2_since_year'].astype( int )
df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )



# 3.0 Data Filtering

df3 = df2.copy()

## 3.1 Rows filtering

df3 = df3[(df3['open'] != 0) & (df3['sales'] > 0)]

## 3.2 Columns filtering

cols_drop = ['customers', 'open', 'promo_interval', 'month_map']
# axis 0 => linhas
# axis 1 => colunas
df3 = df3.drop( cols_drop, axis=1 )


df3.columns


# 4.0 Exploratory Data Analysis ( EDA )

df4 = df3.copy()

## 4.1 Analise Univariada

### 4.1.1 Respose Variable

sns.displot( df4['sales'] )

### 4.1.2 Numerical Variable

plt.figure( figsize=( 50, 5 ) )
num_attributes.hist( bins=25 );

### 4.1.3 Categorical Variable

df4['state_holiday'].drop_duplicates()

plt.figure( figsize=( 22, 15 ) )
# state_holiday
plt.subplot( 3, 3, 1 )
a = df4[df4['state_holiday'] != 'regular_day']
sns.countplot( a['state_holiday'] )

plt.subplot( 3, 3, 3 )
sns.kdeplot( df4[df4['state_holiday'] == 'public_holiday']['sales'], label='public_holiday', shade=True )
sns.kdeplot( df4[df4['state_holiday'] == 'easter_holiday']['sales'], label='easter_holiday', shade=True )
sns.kdeplot( df4[df4['state_holiday'] == 'christmas']['sales'], label='christmas', shade=True )

# store_type
plt.subplot( 3, 3, 4 )
sns.countplot( df4['store_type'] )

plt.subplot( 3, 3, 6 )
sns.kdeplot( df4[df4['store_type'] == 'a']['sales'], label='a', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'b']['sales'], label='b', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'c']['sales'], label='c', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'd']['sales'], label='d', shade=True )

# assortment
plt.subplot( 3, 3, 7 )
sns.countplot( df4['assortment'] )

plt.subplot( 3, 3, 9 )
sns.kdeplot( df4[df4['assortment'] == 'extended']['sales'], label='extended', shade=True )
sns.kdeplot( df4[df4['assortment'] == 'basic']['sales'], label='basic', shade=True )
sns.kdeplot( df4[df4['assortment'] == 'extra']['sales'], label='extra', shade=True )

## 4.2 Analise Bivariada

### **H1.** Stores with larger assortments (product mix) should sell more.
**FALSE** Stores with MORE ASSISTANCE sell LESS.<p>
**Is it important for the model?** Yes, because even if the basic and the extended are similar, the extra has a peculiar behavior that it is worth training the model with it.

aux1 = df4[['assortment','sales']].groupby( 'assortment' ).sum().reset_index()
sns.barplot( x='assortment', y='sales', data=aux1 );

aux2 = df4[['year_week','assortment','sales']].groupby( ['year_week','assortment'] ).sum().reset_index()
aux3 = aux2.pivot( index='year_week', columns='assortment', values='sales' )
aux3.plot()

aux4 = aux2[ aux2['assortment'] == 'extra' ]
aux5 = aux4.pivot( index='year_week', columns='assortment', values='sales' )
aux5.plot()

aux2.head()

### **H2.** Stores with closer competitors should sell less.(common sense)
**FALSE** Stores with CLOSER COMPETITORS sell MORE.<p>
**Is it important for the model?** It might be important for the model, but in a weaker way

plt.figure( figsize=( 10, 7 ) )
aux1 = df4[['competition_distance', 'sales']].groupby( 'competition_distance' ).sum().reset_index()

plt.subplot( 3, 1, 1 )
sns.scatterplot( x ='competition_distance', y='sales', data=aux1 );

plt.subplot( 3, 1, 2 )
bins = list( np.arange( 0, 20000, 1000) )
aux1['competition_distance_binned'] = pd.cut( aux1['competition_distance'], bins=bins )
aux2 = aux1[['competition_distance_binned', 'sales']].groupby( 'competition_distance_binned' ).sum().reset_index()
sns.barplot( x='competition_distance_binned', y='sales', data=aux2 );
plt.xticks( rotation=90 );

plt.subplot( 3, 1, 3 )
x = sns.heatmap( aux1.corr( method='pearson' ), annot=True );# ver a correlaçao das distancias e das vendas
bottom, top = x.get_ylim()
x.set_ylim( bottom+0.5, top-0.5 );

aux1.sample(4)

### **H3.** Stores with longer-term competitors should sell more.
**False** Stores with longer-time competitors sell less, very counterintuitive

**Is it important for the model?** may be, but not so, weak pearson

plt.figure( figsize=( 10, 7 ) )
plt.subplot( 1, 3, 1 )
aux1 = df4[['competition_time_month', 'sales']].groupby( 'competition_time_month' ).sum().reset_index()
aux2 = aux1[( aux1['competition_time_month'] < 120 ) & ( aux1['competition_time_month'] != 0 )]
sns.barplot( x='competition_time_month', y='sales', data=aux2 );
plt.xticks( rotation=90 );

plt.subplot( 1, 3, 2 )
sns.regplot( x='competition_time_month', y='sales', data=aux2 );

plt.subplot( 1, 3, 3 )
x = sns.heatmap( aux1.corr( method='pearson'), annot=True );
bottom, top = x.get_ylim()
x.set_ylim( bottom+0.5, top-0.5);

df4.columns


### H4 .Stores with larger assortments should sell more.
**False** because stores with promotions active for a long time sell less after a certain period of promotion

**Is it important for the model?** No, low pearson

plt.figure( figsize=( 30, 9 ) )

aux1 = df4[['promo_time_week', 'sales']].groupby( 'promo_time_week').sum().reset_index()

grid = plt.GridSpec( 2, 3 )

plt.subplot( grid[0,0] )
aux2 = aux1[aux1['promo_time_week'] > 0] # promo extendido
sns.barplot( x='promo_time_week', y='sales', data=aux2 );
plt.xticks( rotation=90 );

plt.subplot( grid[0,1] )
sns.regplot( x='promo_time_week', y='sales', data=aux2 );

plt.subplot( grid[1,0] )
aux3 = aux1[aux1['promo_time_week'] < 0] # promo regular
sns.barplot( x='promo_time_week', y='sales', data=aux3 );
plt.xticks( rotation=90 );

plt.subplot( grid[1,1] )
sns.regplot( x='promo_time_week', y='sales', data=aux3 );

plt.subplot( grid[:,2] )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

### <s>H5. Stores with more promotion days should sell more</s>

# let's leave it for the second cycle of CRISP-DaS

### H7.Stores with more consecutive promotions should sell more
- This hypothesis is not so relevant for the model, as it is very little different from the traditional and extended to the extended one, and the strange behavior that could be analyzed by the model is very small so it shouldn't even influence, then we get an opinion from an algorithm

**False** Stores with more consecutive promotions sell less

**Is it important for the model?** Yes, there is a phenomenon that the traditional way falls out of nowhere, which can be important for the model to learn

df4[['promo','promo2','sales']].groupby( ['promo','promo2'] ).sum().reset_index()



aux1 = df4[( df4['promo'] == 1 ) & ( df4['promo2'] == 1 )][['year_week', 'sales']].groupby( 'year_week' ).sum().reset_index()
ax = aux1.plot()

aux2 = df4[( df4['promo'] == 1 ) & ( df4['promo2'] == 0 )][['year_week', 'sales']].groupby( 'year_week' ).sum().reset_index()
aux2.plot( ax=ax )

ax.legend( labels=['Tradicional & Extendida', 'Extendida']);


### **H8.** Stores open during the Christmas holiday should sell more.
**False** Stores open during the Christmas holiday sell less

- It's already known, it's not an insight

**Is it important for the model?** Yes, because if it's a holiday it changes how the model has to predict

plt.figure( figsize=( 20, 5 ) )
plt.subplot( 1,2,1 )
aux = df4[ df4['state_holiday'] != 'regular_day' ]
aux1 = aux[['state_holiday', 'sales']].groupby( 'state_holiday' ).sum().reset_index()
sns.barplot( x='state_holiday', y='sales', data=aux1 );

plt.subplot( 1,2,2 )
aux2 = aux[['year','state_holiday', 'sales' ]].groupby( [ 'year', 'state_holiday' ] ).sum().reset_index()
sns.barplot( x='year', y='sales', hue='state_holiday', data=aux2 );

### **H9.** Stores should sell more over the years.
**False** Stores sell less over the years

- It's not insight, because people should already know
- Always compare between closed periods, otherwise there will be a problem in your analysis, since you do not have all the data for that period

**Is it important for the model?** Yes, very high correlation

plt.figure( figsize=( 22, 15 ) )
aux1 = df4[['year', 'sales']].groupby( 'year' ).sum().reset_index()

plt.subplot( 1,3,1 )
sns.barplot( x='year', y='sales', data=aux1 );

plt.subplot( 1,3,2 )
sns.regplot( x='year', y='sales', data=aux1 );

plt.subplot( 1,3,3 )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

### **H10.** Stores should sell more in the second half of the year.

- It's not an insight, people should already know that, it's neither counter-intuitive nor new information


**Is it important for the model?**Yes, high correlation

plt.figure( figsize=( 22, 15 ) )
aux1 = df4[['year', 'sales']].groupby( 'year' ).sum().reset_index()

plt.subplot( 1,3,1 )
sns.barplot( x='year', y='sales', data=aux1 );

plt.subplot( 1,3,2 )
sns.regplot( x='year', y='sales', data=aux1 );

plt.subplot( 1,3,3 )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

### **H11.** Stores should sell more after the 10th of each month.
**True** Stores sell more after the 10th of each month

- It's not insight, because people follow it already

**Is it important for the model?** Yes, because it has a certain negative correlation (the larger one, the smaller the other)

plt.figure( figsize=( 22, 15 ) )
aux1 = df4[['day', 'sales']].groupby( 'day' ).sum().reset_index()

plt.subplot( 2,2,1 )
sns.barplot( x='day', y='sales', data=aux1 );

plt.subplot( 2,2,2 )
sns.regplot( x='day', y='sales', data=aux1 );

plt.subplot( 2,2,3 )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

aux1['before_after'] = aux1['day'].apply( lambda x: 'before_10_days' if x<=10 else 'after_10_days' )

plt.subplot( 2,2,4 )
aux2 = aux1[['before_after','sales']].groupby( 'before_after' ).sum().reset_index()
sns.barplot( x='before_after', y='sales', data=aux2 );


### **H12.** Stores should sell less on weekends.
**True** Stores sell less on weekends

**Is it important for the model?** Yes, high correlation

plt.figure( figsize=( 22, 15 ) )
aux1 = df4[['day_of_week', 'sales']].groupby( 'day_of_week' ).sum().reset_index()

plt.subplot( 1,3,1 )
sns.barplot( x='day_of_week', y='sales', data=aux1 );

plt.subplot( 1,3,2 )
sns.regplot( x='day_of_week', y='sales', data=aux1 );

plt.subplot( 1,3,3 )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

### **H13.** Stores should sell less during school holidays.
**True** Stores sell less during school holidays, except July and August

- Noa should be insight, the business team should already know that

**Is it important for the model?** Yes, to model the algorithm you need to know if it's a school holiday, and what month it is

plt.figure( figsize=( 22, 15 ) )
aux1 = df4[['school_holiday', 'sales']].groupby( 'school_holiday' ).sum().reset_index()

plt.subplot( 2,1,1 )
sns.barplot( x='school_holiday', y='sales', data=aux1 );

aux2 = df4[['month','school_holiday', 'sales']].groupby([ 'month','school_holiday' ]).sum().reset_index()

plt.subplot( 2,1,2 )
sns.barplot( x='month', y='sales', hue='school_holiday', data=aux2 );


## 4.2.1 Summary of hypotheses

# from tabulate import tabulate


# tab =[['Hipoteses', 'Conclusao', 'Relevancia'],
#       ['H1', 'Falsa', 'Baixa'],  
#       ['H2', 'Falsa', 'Media'],  
#       ['H3', 'Falsa', 'Media'],
#       ['H4', 'Falsa', 'Baixa'],
#       ['H5', '-', '-'],
#       ['H7', 'Falsa', 'Baixa'],
#       ['H8', 'Falsa', 'Media'],
#       ['H9', 'Falsa', 'Alta'],
#       ['H10', 'Falsa', 'Alta'],
#       ['H11', 'Verdadeira', 'Alta'],
#       ['H12', 'Verdadeira', 'Alta'],
#       ['H13', 'Verdadeira', 'Baixa'],
#      ]  
# print( tabulate( tab, headers='firstrow' ) )


## 4.3 Multivariate Analysis

### 4.3.1 Numerical Attributes

correlation = num_attributes.corr( method='pearson' )
plt.figure( figsize=( 25, 19 ) )
sns.heatmap( correlation, annot=True );

### 4.3.2 Categorical Attributes

a = df4.select_dtypes( include='object' )

a.head()

cm = pd.crosstab( a['state_holiday'], a['store_type'] )
cm = cm.values

a1 = cramer_v(a['state_holiday'], a['state_holiday'])
a2 = cramer_v(a['state_holiday'], a['store_type'])
a3 = cramer_v(a['state_holiday'], a['assortment'])

a4 = cramer_v(a['store_type'], a['state_holiday'])
a5 = cramer_v(a['store_type'], a['store_type'])
a6 = cramer_v(a['store_type'], a['assortment'])

a7 = cramer_v(a['assortment'], a['state_holiday'])
a8 = cramer_v(a['assortment'], a['store_type'])
a9 = cramer_v(a['assortment'], a['assortment'])


d = pd.DataFrame( {'state_holiday': [a1,a2,a3],
                   'store': [a4,a5,a6],
                   'assortment': [a7,a8,a9],    
                    })
d = d.set_index( d.columns )


d

plt.figure( figsize=( 17, 10 ) )
sns.heatmap( d, annot=True )


# 5.0 Data Preparation

df5 = df4.copy()

## 5.1 Standardization

# It is not necessary to do so, as there is no categorical variable
# has a distribution that looks normal by itself

## 5.2 Rescaling

a = df5.select_dtypes( include=['int64', 'float64','int32','UInt32'] )

plt.figure( figsize=( 17, 10 ) )
sns.boxplot( df5['competition_distance'] )

rs = RobustScaler()
mms = MinMaxScaler()

# competition distance
df5['competition_distance'] = rs.fit_transform( df5[['competition_distance']].values )
pickle.dump( rs, open( 'parameters/competition_distance_scaler.pkl', 'wb') )

# competition time month
df5['competition_time_month'] = rs.fit_transform( df5[['competition_time_month']].values )
pickle.dump( rs, open( 'parameters/competition_time_month_scaler.pkl', 'wb') )

# promo time week
df5['promo_time_week'] = mms.fit_transform( df5[['promo_time_week']].values )
pickle.dump( rs, open( 'parameters/promo_time_week_scaler.pkl', 'wb') )

# year
df5['year'] = mms.fit_transform( df5[['year']].values )
pickle.dump( mms, open( 'parameters/year_scaler.pkl', 'wb') )

plt.figure( figsize=( 17, 10 ) )
sns.distplot( df5['competition_distance'] );

## 5.3 Transformation

### 5.3.1 Encoding

df5.head()

# state holiday
df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

# store type
le = LabelEncoder()
df5['store_type'] = le.fit_transform( df5['store_type'] )
pickle.dump( le, open( 'C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/parameters/store_type_scaler.pkl', 'wb') )


df5['assortment'].drop_duplicates() 
# assortment
assortment_dict = { 'basic':1, 'extra':2, 'extended':3 }
df5['assortment'] =  df5['assortment'].map( assortment_dict )

### 5.3.2 Response Variable Transformation

df5['sales'] = np.log1p( df5['sales'] )

sns.displot( df5['sales'] )

### 5.3.3 Nature Transformation( cycle transformation )

# month
df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2 * np.pi/12 ) ) )
df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2 * np.pi/12 ) ) )

# day
df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2 * np.pi/30 ) ) )
df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2 * np.pi/30 ) ) )

# Week of year
df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2 * np.pi/52 ) ) )
df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2 * np.pi/52 ) ) )                                                 
                                                    
# day of week
df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2 * np.pi/7 ) ) )
df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2 * np.pi/7 ) ) )                                                                    

# 6.0 Feature Selection


df6 = df5.copy()

## 6.1 Split dataframe into training and test dataset

cols_drop = ['week_of_year','day','month', 'day_of_week','promo_since','competition_since', 'year_week' ]
df6 = df6.drop( cols_drop, axis=1 )

df6[['store','date']].groupby( 'store' ).max().reset_index()['date'][0]  - datetime.timedelta( days=6*7 )

# training dataset
X_train = df6[df6['date'] < '2015-06-19'] 
y_train = X_train['sales']

# test dataset
X_test = df6[df6['date'] >= '2015-06-19']
y_test = X_test['sales']

print( 'Training Min Date: {}'.format( X_train['date'].min() ) )
print( 'Training Max Date: {}'.format( X_train['date'].max() ) )

print( '\nTest Min Date: {}'.format( X_test['date'].min() ) )
print( 'Test Max Date: {}'.format( X_test['date'].max() ) )

## 6.2 Boruta as Feature Selector

# training and test dataset for Boruta
X_train_n = X_train.drop( ['date', 'sales'], axis=1 ).values  
y_train_n = y_train.values.ravel() 

# define RandomForestRegressor
rf = RandomForestRegressor( n_jobs=-1 ) 

# define Boruta
boruta = BorutaPy( rf, n_estimators='auto', verbose=2, random_state=42 ).fit( X_train_n, y_train_n ) 

### 6.2.1 Best Features From Boruta 


cols_selected = boruta.support_.tolist() 

# best features 
x_train_fs = x_train.drop( ['date', 'sales'], axis=1 ) 
cols_selected_boruta = x_train.iloc[:, cols_selected].columns.to_list

# not selected boruta 
cols_not_selected_boruta = np.setdiff1d( x_train_fs.columns, cols_selected_boruta )


### 6.2.2 Manual feature selection

cols_selected_boruta = [
    'store',
    'promo',
    'store_type',
    'assortment',
    'competition_distance',
    'competition_open_since_month',
    'competition_open_since_year',
    'promo2',
    'promo2_since_week',
    'promo2_since_year',
    'competition_time_month',
    'promo_time_week',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'day_sin',
    'day_cos',
    'week_of_year_sin',
    'week_of_year_cos']

# columns to add
feat_to_add = ['date', 'sales']

cols_selected_boruta_full = cols_selected_boruta.copy()
cols_selected_boruta_full.extend( feat_to_add )

# 7.0 ML models 

x_train = X_train[ cols_selected_boruta ]  
x_test = X_test[ cols_selected_boruta ]

x_training  = X_train[ cols_selected_boruta_full ] 

## 7.1 Average Model

aux1 = x_test.copy()

aux1['sales'] = y_test.copy()

#prediction
aux2 = aux1[['store', 'sales']].groupby( 'store' ).mean().reset_index().rename( columns={'sales':'predictions'}  )
aux1 = pd.merge( aux1, aux2, how='left', on='store' )
yhat_baseline = aux1['predictions'] 

#performance
baseline_result = ml_error( 'Average Model', np.expm1( y_test ), np.expm1( yhat_baseline ) ) 
baseline_result


# cross validation n faz sentido fazer pra esse modelo

## 7.2 Linear Regression Model

# model
lr = LinearRegression().fit( x_train, y_train )


# prediction 
yhat_lr = lr.predict( x_test )

# performance 
lr_result = ml_error( 'Linear Regression', np.expm1( y_test ), np.expm1( yhat_lr ) )
lr_result 

### 7.2.1 Linear Regression Model Crossvalidation

lr_result_cv = cross_validation( x_training, 5, 'Linear Regression', lr, verbose=False ) # 5 kfols => 5 separaçaoes do nosso dado de treinamento, verbose=False => significa que nao queremos que ele fique printando as iteraçoes

lr_result_cv

## 7.3 Linear Regression Regularized Model - Lasso


# model
lrr = Lasso( alpha=0.01 ).fit( x_train, y_train )


# prediction 
yhat_lrr = lrr.predict( x_test )

# performance 
lrr_result = ml_error( 'Linear Regression - Lasso', np.expm1( y_test ), np.expm1( yhat_lrr ) )
lrr_result

### 7.3.1 Linear Regression Regularized Model - Lasso Cross validation


lrr_result_cv = cross_validation( x_training, 5, 'Lasso', lrr, verbose=False ) 

lrr_result_cv

## 7.4 Random Forest Regressor

# model
rf = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=42 ).fit( x_train, y_train )

# prediction 
yhat_rf = rf.predict( x_test )

# performance 
rf_result = ml_error( 'Random forest Regressor', np.expm1( y_test ), np.expm1( yhat_rf ) )
rf_result



### 7.4.1 Random Forest Regressor CrossValidation

# rf_result_cv = cross_validation( x_training, 5, 'Random Forest Regressor', rf, verbose=True )
# rf_result_cv

# "Random Forest Regressor	837.68 +/- 219.1	0.12 +/- 0.02	1256.08 +/- 320.36"
# rf_result_cv
#  demora 11h pra executar o codigo de crossvalidation com random forest +/-
print ( "Random Forest Regressor	837.68 +/- 219.1	0.12 +/- 0.02	1256.08 +/- 320.36" )

## 7.5 XGBoost Regressor

# model
model_xgb = xgb.XGBRegressor( objective='reg:squarederror',
                              eta=0.01, 
                              n_estimators=100, 
                              n_jobs=-1, 
                              max_depth=10,
                              subsample=0.7,
                              colsample_bytee=0.9, 
                              ).fit( x_train, y_train ) 




# prediction 
yhat_xgb = model_xgb.predict( x_test )

# performance 
xgb_result = ml_error( 'XGB Regressor', np.expm1( y_test ), np.expm1( yhat_xgb ) )
xgb_result

## 7.5.1 XGBoost Regressor Crossvalidation

xgb_result_cv = cross_validation( x_training, 5, 'XGBoost Regressor', model_xgb, verbose=True )
xgb_result_cv
# XGBoost Regressor	1030.28 +/- 167.19	0.14 +/- 0.02	1478.26 +/- 229.79

## 7.6 Compare Model's Performance

### 7.6.1 Compare Model's Performance - Single Performance

modelling_result = pd.concat( [baseline_result, lr_result, lrr_result, rf_result, xgb_result] )
modelling_result.sort_values( 'RMSE' )

    # modelname            mae          mape        rmse
# 0	Random forest Regressor	679.598831	0.099913	1011.119437
# 0	Average Model	1354.800353	0.206400	1835.135542
# 0	Linear Regression	1863.879857	0.292198	2672.592495
# 0	Linear Regression - Lasso	1890.157813	0.288960	2745.550239
# 0	XGB Regressor	6683.606400	0.949503	7330.742181

### 7.6.2 Compare Model's Performance -  Real Performance

modelling_result = pd.concat( [lr_result_cv, lrr_result_cv, rf_result, xgb_result] )
modelling_result.sort_values( 'RMSE' )

#   model-name           MAE CV               MAPE CV           RMSE CV
# 0	Linear Regression	2081.73 +/- 295.63	 0.3 +/- 0.02	 2952.52 +/- 468.37
# 0	Lasso	            2116.38 +/- 341.5	 0.29 +/- 0.01	 3057.75 +/- 504.26
# 0	Random Forest Regr	837.68 +/- 219.1	 0.12 +/- 0.02	 1256.08 +/- 320.36
# 0	XGBoost Regressor	1030.28 +/- 167.19	 0.14 +/- 0.02	 1478.26 +/- 229.79


# 8.0 Hyperparameter fine tuning

## 8.1 Random Search

param = {
   'n_estimators': [15, 17, 25, 30, 35],
   'eta': [0.01, 0.03],
   'max_depth': [3, 5, 9],
   'subsample': [0.1, 0.5, 0.7],
   'colsample_bytree': [0.3, 0.7, 0.9],
   'min_child_weight': [3, 8, 15]
       }

MAX_EVAL = 10 

# choose values for parameters randomly
hp = { k: random.sample( v, 1 )[0] for k, v in param.items() }
print( hp )
   
# model
model_xgb = xgb.XGBRegressor( objective='reg:squarederror',
                                 n_estimators=hp['n_estimators'], 
                                 eta=hp['eta'], 
                                 max_depth=hp['max_depth'], 
                                 subsample=hp['subsample'],
                                 colsample_bytee=hp['colsample_bytree'],
                                 min_child_weight=hp['min_child_weight'] )

# performance
result = cross_validation( x_training, 5, 'XGBoost Regressor', model_xgb, verbose=True )
final_result = pd.concat( [final_result, result] )
       
final_result

## 8.2 Final Model 

param_tuned = {
    'n_estimators': 3000,
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3 
        }

# model
model_xgb_tuned = xgb.XGBRegressor( objective='reg:squarederror',
                                    n_estimators=param_tuned['n_estimators'], 
                                    eta=param_tuned['eta'], 
                                    max_depth=param_tuned['max_depth'], 
                                    subsample=param_tuned['subsample'],
                                    colsample_bytree=param_tuned['colsample_bytree'],
                                    min_child_weight=param_tuned['min_child_weight'] ).fit( x_train, y_train ) # pois o ml_error so ve a performance do modelo, mas n o treina

# prediction
yhat_xgb_tuned = model_xgb_tuned.predict( x_test )  

# performance
xgb_result_tuned = ml_error( 'XGBoost Regressor', np.expm1( y_test ), np.expm1( yhat_xgb_tuned ) )
xgb_result_tuned



pickle.dump( model_xgb_tuned, open( 'C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/parameters/model_rossmann.pkl', 'wb') )

#                         mae         mape        rmse
# 0	XGBoost Regressor	664.974996	0.097529	957.774225


mpe = mean_percentage_error( np.expm1( y_test ), np.expm1( yhat_xgb_tuned ) )

# 9.0 Translation and interpretation of the error

df9 = X_test[ cols_selected_boruta_full ]

# rescale
df9['sales'] = np.expm1( df9['sales'] )
df9['predictions'] = np.expm1( yhat_xgb_tuned )


## 9.1 Business Performance

# sum of predictions
df91= df9[['store','predictions']].groupby( 'store' ).sum().reset_index()  

# MAE and MAPE
df9_aux1 = df9[['store', 'sales', 'predictions']].groupby('store').apply( lambda x: mean_absolute_error( x['sales'], x['predictions'] ) ).reset_index(). rename( colums={0:'MAE'} )


df9_aux2 = df9[['store', 'sales', 'predictions']].groupby('store').apply( lambda x: mean_absolute_percentage_error( x['sales'], x['predictions'] ) ).reset_index().rename( colums={0:'MAPE'} )

# Merge
df9_aux3 = pd.merge( df9_aux1, df9_aux2, how='inner', on='store' )
df92 =  pd.merge( df91, df9_aux3, how='inner', on='store' )

# Scenarios 
df92['worst_scenario'] = df92['predictions'] - df92['MAE']
df92['best_scenario'] = df92['predictions'] + df92['MAE']

# ordering columns
df92 = df92[['store', 'predictions', 'worst_scenario', 'best_scenario', 'MAE', 'MAPE']]


df92.sort_values( 'MAPE', ascending=False ).head() 

sns.scatterplot( x='store', y='MAPE', data=df92 )

## 9.2 Total Performance

df93 = df92[['predictions', 'worst_scenario', 'best_scenario']].apply( lambda x: np.sum( x ), axis=0 ).reset_index().rename( columns={'index': 'Scenario', 0:'Values'} )
df93['Values'] = df93['Values'].map( 'R${:,.2f}'.format )
df93

## 9.3 Machine Learning Performance

df9['error'] = df9['sales'] - df9['predictions']
df9['error_rate'] = df9['predictions'] / df9['sales']

plt.subplot( 2, 2, 1 )
sns.lineplot( x='date', y='sales', data=df9, label='Sales' )
sns.lineplot( x='date', y='predictions', data=df9, label='Predictions' )

plt.subplot( 2, 2, 2 )
sns.lineplot( x='date', y='error_rate', data=df9 )
plt.axhline( 1, linestyle='--' ) # plotar uma linha no 1 atraves do eixo x

plt.subplot( 2, 2, 3 ) 
sns.distplot( df9['error'] ) # nossos erros tem um distribuiçao perto da gaussiana, a normal, isso é mt bom

plt.subplot( 2, 2, 4 )
sns.scatterplot( df9['predictions'], df9['error'] )


# 10.0 Deploy 

## 10.1 Rossmann Class

import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path='C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/'
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb') )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb') )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb') )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameters/year_scaler.pkl', 'rb') )
        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameters/store_type_scaler.pkl', 'rb') )
        
        
    def data_cleaning( self, df1 ): 
        
        ## 1.1. Rename Columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        # rename
        df1.columns = cols_new

        ## 1.3. Data Types
        df1['date'] = pd.to_datetime( df1['date'] )

        ## 1.5. Fillout NA
        #competition_distance        
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )

        #competition_open_since_month
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )

        #competition_open_since_year 
        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )

        #promo2_since_week           
        df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )

        #promo2_since_year           
        df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )

        #promo_interval              
        month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['promo_interval'].fillna(0, inplace=True )

        df1['month_map'] = df1['date'].dt.month.map( month_map )

        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )

        ## 1.6. Change Data Types
        # competiton
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
        
        return df1 


    def feature_engineering( self, df2 ):

        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # competition since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
        df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

        # assortment
        df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
        ## 3.1. Filtragem das Linhas
        df2 = df2[df2['open'] != 0]

        ## 3.2. Selecao das Colunas
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop( cols_drop, axis=1 )
        
        return df2


    def data_preparation( self, df5 ):

        ## 5.2. Rescaling 
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )
    
        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values )

        # promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform( df5[['promo_time_week']].values )
        
        # year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )

        ### 5.3.1. Encoding
        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map( assortment_dict )

        
        ### 5.3.3. Nature Transformation
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day 
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        
        cols_selected = [ 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
            'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']
        
        return df5[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )
        

## 10.2 API Handler

from flask import Flask, request, Response # flask é o modulo e Flask é a classe
import pickle
import pandas as pd
from rossmann.Rossmann import Rossmann

# loading model 
model = pickle.load( open( 'C:/Users/PICHAU/Desktop/AnaliseDeDados/DsEmProd/parameters/model_rossmann.pkl', 'rb') ) # carregando o modelo em memoria

# initialize API
app = Flask( __name__ )

# criar rotas que vai receber requests, onde fica o endpoint 
@app.route( 'rossmann/predicts', methods=['POST'] )# metodos que ele envia algum dado para poder receber

def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class,  criar uma copia da classe 
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
        
    else:
        return Reponse( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '0.0.0.0' ) # esse numero quer dizer local host, esse app esta rodando na minha maquina
    

## 10.3 API Tester(send data to api)

# loading test dataset 
df10 = pd.read_csv( 'datasets/test.csv' )

df10

# merge test dataset + store
df_test = pd.merge( df10, df_store_raw, how='left', on='Store'  )

# choose especific store for prediction
df_test = df_test[df_test['Store'].isin( [7, 3, 9] )]

# remove closed days
df_test = df_test[ df_test['Open'] != 0 ]
df_test = df_test[ ~df_test['Open'].isnull() ]
df_test = df_test.drop( 'Id', axis=1 )




# convert dataframe to json
data = json.dumps( df_test.to_dict( orient='records' ) )


data

# API Call
# url = 'http://192.168.0.106:5000/rossmann/predict'
url = 'https://prediction-rossmann-v2.herokuapp.com/rossmann/predict'
header = {'Content-type': 'application/json' } 
data = data

r = requests.post( url, data=data, headers=header )
print( 'Status Code {}'.format( r.status_code ) )

d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )

d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()

for i in range( len( d2 ) ):
    print( 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format( 
            d2.loc[i, 'store'], 
            d2.loc[i, 'prediction'] ) )