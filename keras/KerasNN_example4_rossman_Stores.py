# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:50:04 2019
Source: [Jojo_John_Moolayil]_Learn_Keras_for_Deep_Neural_N.pdf
@author: chrisxt
Objective: DNN model used for regression
"""
#----------------------------------------------------------------------------------------------------------------
#Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Load datasets
df = pd.read_csv("/home/chrisxt/Documents/MachineLearning/keras/datasets/rossmann-store-sales/train.csv")

print("Shape of the Dataset:",df.shape)
#the head method displays the first 5 rows of the data
df.head(5)

store = pd.read_csv("/home/chrisxt/Documents/MachineLearning/keras/datasets/rossmann-store-sales/store.csv")
print("Shape of the Dataset:",store.shape)
#Display the first 5 rows of data using the head method of pandas dataframe
store.head(5)

#----------------------------------------------------------------------------------------------------------------
#Information from datasets
#train.csv
#Store: a unique ID for each store
#Sales: the turnover for a given day (our target y variable)
#Customers: the number of customers on a given day
#Open: an indicator for whether the store was open: 0 = closed, 1 = open
#StateHoliday: indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. 
#   Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = none
#SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools

#store.csv
#StoreType: differentiates between four different store models: a, b, c, d
#Assortment: describes an assortment level: a = basic, b = extra, c = extended
#CompetitionDistance: distance in meters to the nearest competitor store
#CompetitionOpenSince[Month/Year]: gives the approximate year and month of the time the nearest competitor was opened
#Promo: indicates whether a store is running a promo on that day 
#Promo2: Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
#Promo2Since[Year/Week]: describes the year and calendar week when the store started participating in Promo2
#PromoInterval: describes the consecutive intervals at which Promo2 is started, naming the months the
#   promotion is started anew (e.g., “Feb, May, Aug, Nov” means each round starts in February, May, August, and November of any given year for that store)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Joins the train and store dataframe to create a new dataframe.
df_new = df.merge(store,on=["Store"], how="inner")
print(df_new.shape)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#exploring the dataset
print("Distinct number of Stores :", len(df_new["Store"].unique()))
print("Distinct number of Days :", len(df_new["Date"].unique()))
print("Average daily sales of all stores : ",round(df_new["Sales"].mean(),2))
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Finding Data Types
df_new.dtypes
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Working with Time
df_new["DayOfWeek"].value_counts()
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Create additional features that will help our model learn patterns better. 
#We will create the week number, month, day, quarter, and year as features from the date variable.
#Extract all date properties from a datetime datatype:
df_new['Date'] = pd.to_datetime(df_new['Date'], infer_datetime_format=True)
df_new["Month"] = df_new["Date"].dt.month
df_new["Quarter"] = df_new["Date"].dt.quarter
df_new["Year"] = df_new["Date"].dt.year
df_new["Day"] = df_new["Date"].dt.day
df_new["Week"] = df_new["Date"].dt.week
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Add a new feature based on climate and seasons
df_new["Season"] = np.where(df_new["Month"].isin([3,4,5]),"Spring", np.where(df_new["Month"].isin([6,7,8]),"Summer", np.where(df_new["Month"].isin([9,10,11]),"Fall", np.where(df_new["Month"].isin([12,1,2]),"Winter","None"))))

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#View new features
print(df_new[["Date","Year","Month","Day","Week","Quarter","Season"]].head())
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Visualize data
plt.figure(figsize=(15,8))
plt.hist(df_new["Sales"])
plt.title("Histogram for Store Sales")
plt.xlabel("bins")
plt.xlabel("Frequency")
plt.show()
#Explanation:
#The histogram helps us understand the distribution of the data at a high level. 
#From the preceding plot, we can see that the data range is from 0 to 40,000,
#but there is barely any data after 20,000. This indicates that most of the stores 
#have sales in the range 0–20,000, and just a few stores have sales greater than 20,000. 
#It might be worthwhile to remove these outliers, as it helps the model learn better.
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Exploring Numeric Columns
df_new.hist(figsize=(20,10))
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Let’s have a look at the number of missing data points in each column (if any) in its associated percentage form.
#As a rule of thumb, if there is a loss of anything between 0% and 10%, we can make a few attempts to fill the 
#missing points and use the feature.
df_new.isnull().sum()/df_new.shape[0] * 100
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Filling missing data
#Replace nulls with the mode
df_new["CompetitionDistance"]=df_new["CompetitionDistance"].fillna(df_new["CompetitionDistance"].mode()[0])
#Double check if we still see nulls for the column
df_new["CompetitionDistance"].isnull().sum()/df_new.shape[0] * 100
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Categorical Features
#The best way to study a categorical variable is to study the impact on the target variable from its individual classes.
sns.set(style="whitegrid")
#Create the bar plot for Average Sales across different Seasons
ax = sns.barplot(x="Season", y="Sales", data=df_new)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Create the bar plot for Average Sales across different Assortments
ax = sns.barplot(x="Assortment", y="Sales", data=df_new)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Create the bar plot for Average Sales across different Store Types
ax = sns.barplot(x="StoreType", y="Sales", data=df_new)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#bar plots for the same set of categorical variables we studied earlier, albeit for counts
ax = sns.barplot(x="Season", y="Sales", data=df_new,estimator=np.size)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
ax = sns.barplot(x="Assortment", y="Sales", data=df_new,estimator=np.size)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
ax = sns.barplot(x="StoreType", y="Sales", data=df_new,estimator=np.size)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#all categorical features stored as text columns need to be converted to a one-hot encoded form for the model training data
target = ["Sales"]
numeric_columns = ["Customers","Open","Promo","Promo2","StateHoliday","SchoolHoliday","CompetitionDistance"]
categorical_columns = ["DayOfWeek","Quarter","Month","Year","StoreType","Assortment","Season"]
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Define a function that will intake the raw dataframe and the column name and return a one hot encoded DF
def create_ohe(df_new, col):
    le = LabelEncoder()
    a=le.fit_transform(df_new[col]).reshape(-1,1)
    ohe = OneHotEncoder(sparse=False)
    column_names = [col+ "_"+ str(i) for i in le.classes_]
    return (pd.DataFrame(ohe.fit_transform(a),columns =column_names))
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Since the above function converts the column, one at a time
#We create a loop to create the final dataset with all features
temp = df_new[numeric_columns]
for column in categorical_columns:
    temp_df = create_ohe(df_new,column)
    temp = pd.concat([temp,temp_df],axis=1)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Check data
print("Shape of Data:",temp.shape)
print("Distinct Datatypes:",temp.dtypes.unique())
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Check if a column has object as the data type within our dataframe. Objective: All of them should be numeric.
print(temp.columns[temp.dtypes=="object"])

temp["StateHoliday"].unique()
#Output: array(['0', 'a', 'b', 'c', 0], dtype=object)
#The feature seems to have incorrect values. Ideally, StateHoliday
#should have either a 0 or 1 as the possible values to indicate whether it is a
#holiday or not. Let’s repair the feature by replacing all values of “a,” “b,” and
#“c” with 1 and the rest with 0, therefore converting the variable as numeric.
temp["StateHoliday"]= np.where(temp["StateHoliday"]== '0',0,1)
#One last check of the data type
temp.dtypes.unique()
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Create train and test dataset with an 80:20 split
x_train, x_test, y_train, y_test = train_test_split(temp,df_new[target],test_size=0.2,random_state=2018)

#Further divide training dataset into train and validation dataset with an 90:10 split
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=2018)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Check the sizes of all newly created datasets
print("Shape of x_train:",x_train.shape)
print("Shape of y_train:",y_train.shape)
print("Shape of x_val:",x_val.shape)
print("Shape of y_val:",y_val.shape)
print("Shape of x_test:",x_test.shape)
print("Shape of y_test:",y_test.shape)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Defining Model Baseline Performance
#if we assume the mean value of sales in the training dataset to be the prediction for all samples in the test dataset, we
#would have a basic benchmark score. The DL model should at least score better than this score to be considered as useful.
#The metric we shall use to perform this test is MAE (mean absolute error).

#calculate the average score of the train dataset
mean_sales = y_train.mean()
print("Average Sales :",mean_sales)

#Now, if we assume the average sales as the prediction for all samples in the test dataset, what does the MAE metric look like?
#Calculate the Mean Absolute Error on the test dataset
print("MAE for Test Data:",abs(y_test - mean_sales).mean()[0])
#Benchmark: MAE for Test Data: 2883.587604303127
#If our DL model doesn’t deliver results better (i.e., lower) than the baseline score, then it would barely add any value.
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#****************************************************************************************************************
#                                            Designing the DNN guidelines.
#****************************************************************************************************************
#-Rule 1: Start with small architectures.
#
#    In the case of DNNs, it is always advised to start with a single-layer network with around 100–300 neurons.
#    Train the network and measure performance using the defined metrics (while defining the baseline score). 
#    If the results are not encouraging, try adding one more layer with the same number of neurons and repeating the process.
#
#- Rule 2: When small architectures (with two layers) fail, increase the size.
#
#    When the results from small networks are not great, you need to increase the number of neurons
#    in layers by three to five times (i.e., around 1,000 neurons in each layer). Also, increase regularization
#    to 0.3, 0.4, or 0.5 for both layers and repeat the process for training and performance measurement. 
#
#-Rule 3: When larger networks with two layers fail, go deeper.
#    
#    Try increasing the depth of the network with more and more layers while maintaining regularization with
#    dropout layers (if required) after each dense (or your selected layer) with a dropout rate between 0.2 and 0.5.
#
#-Rule 4: When larger and deeper networks also fail, go even larger and even deeper.
#
#    In case large networks with ~1000 neurons and five or six layers also don’t give the desired performance,try increasing 
#    the width and depth of the network. Try adding layers with 8,000–10,000 neurons per layer and a dropout of 0.6 to 0.8.
#	
#-Rule 5: When everything fails, revisit the data.
#
#    If all the aforementioned rules fail, revisit the data for improved feature engineering and
#    normalization, and then you will need to try other ML alternatives.
#*****************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Create Deep Neural Network Architecture
model = Sequential()
model.add(Dense(150,input_dim = 44,activation="relu")) #The input_dim =44, since the width of the training data=44
model.add(Dense(1,activation = "linear"))

#Configure the model
model.compile(optimizer='adam',loss="mean_absolute_error",metrics=["mean_absolute_error"])

#Train the model
model.fit(x_train.values,y_train.values, validation_data= (x_val,y_val),epochs=10,batch_size=64)

#Benchmark: MAE for Test Data: 2883.587604303127
#Model: val_mean_absolute_error: 706.7125

#Testing the Model Performance
#Use the model's evaluate method to predict and evaluate the test datasets
result = model.evaluate(x_test.values,y_test.values)

#Print the results
for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))

#Metric  loss : 702.16
#Metric  mean_absolute_error : 702.16

#We got a relatively consistent performance on the test dataset too.
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Improving the Model
#Create Deep Neural Network Architecture
model = Sequential()

model.add(Dense(150,input_dim = 44,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(1,activation = "linear"))

#Compile the model
model.compile(optimizer='adam',loss="mean_squared_error",metrics=["mean_absolute_error"])

#Train the model
history = model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=10,batch_size=64)

#Testing the Model Performance
result = model.evaluate(x_test,y_test)

#Print the results
for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))
     
#    Metric  mean_absolute_error : 646.8
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#New deeper model
model = Sequential()
model.add(Dense(150,input_dim = 44,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(1,activation = "linear"))
model.compile(optimizer='adam',loss="mean_squared_error",metrics=["mean_absolute_error"])
model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=15,batch_size=64)

result = model.evaluate(x_test,y_test)

for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))

#Metric  mean_absolute_error : 635.84
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
#Increasing the Number of Neurons
model = Sequential()
model.add(Dense(350,input_dim = 44,activation="relu"))
model.add(Dense(350,activation="relu"))
model.add(Dense(1,activation = "linear"))
model.compile(optimizer='adam',loss="mean_squared_error",metrics=["mean_absolute_error"])
model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=15,batch_size=64)

result = model.evaluate(x_test,y_test)

for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))

#Metric  mean_absolute_error : 623.83
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Deeper and with more neurons
#added history capability

history = History()
model = Sequential()
model.add(Dense(350,input_dim = 44,activation="relu"))
model.add(Dense(350,activation="relu"))
model.add(Dense(350,activation="relu"))
model.add(Dense(350,activation="relu"))
model.add(Dense(350,activation="relu"))
model.add(Dense(1,activation = "linear"))
model.compile(optimizer='adam',loss="mean_squared_error",metrics=["mean_absolute_error"])
model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=15,batch_size=64,callbacks=[history])

result = model.evaluate(x_test,y_test)

for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))

#Metric  mean_absolute_error : 612.01
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Plotting the Loss Metric Across Epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model's Training & Validation loss across epochs")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

#Once you have finalized the architecture for your model, you can increase the number of epochs for training
#and check if there was any further improvement
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Testing the Model Manually
#Manually predicting from the model, instead of using model's evaluate function
y_test["Prediction"] = model.predict(x_test)
y_test.columns = ["Actual Sales","Predicted Sales"]
print(y_test.head(10))

#Manually predicting from the model, instead of using model's evaluate function
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE :",mean_squared_error(y_test["Actual Sales"].values,y_test["Predicted Sales"].values))
print("MAE :",mean_absolute_error(y_test["Actual Sales"].values,y_test["Predicted Sales"].values))

#MSE : 825525.5321821237
#MAE : 612.0117530558458
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------