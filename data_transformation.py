import pandas as pd
import numpy as np

#This script builds to tables intended for statitistical purposes:
#1)User table with spending per each channel
#2)Summary time table with spending per each channel in each date

#Read data
conversions = pd.read_csv("data/table_A_conversions.txt")
attributions = pd.read_csv("data/table_B_attribution.txt")

#Remove na's 
conversions = conversions.dropna()
attributions = attributions.dropna()
#conversions = conversions.iloc[0:1000,].dropna() #reduced df for test
#attributions = attributions.iloc[0:10000,].dropna() #reduced df for test

#Merge tables - Perform inner join (auto-eliminate nans)
attribs_convs = pd.merge(attributions, conversions, how = "inner", on = "Conv_ID")

##########
#TABLE (1)
##########

#Function to compute user revenue per channel, taking into account IHC index; 
#merged table version
def user_chan_revenue(User_ID, Channel):
	score = 0
	#Sub data frame of interest
	subdf = attribs_convs[(attribs_convs["User_ID"] == User_ID) & 
	(attribs_convs["Channel"] == Channel)]

	if not(subdf.empty):
		score = (subdf["IHC_Conv"]*subdf["Revenue"]).sum()
	
	return score

#Build DataFrame of Interest

#List of user IDs
user_list = list(attribs_convs["User_ID"].unique())

#Extract channels used by users; sort them; cast as list
channel_list = list(attribs_convs["Channel"].sort_values().unique()) 

#Use a matrix to speed up computations
n = len(user_list)
p = len(channel_list)
user_data = np.zeros((n, p)) #NP matrix to be filled with values

#Fill matrix - Computationally expensive
for i in range(0, n):
	#print("Inserting user ", i+1, "/",n) #To roughly evaluate execution time
	for j in range(0, p):
		user_data[i][j] = user_chan_revenue(user_list[i], channel_list[j])

#Convert matrix to dataframe
user_table = pd.DataFrame(user_data, columns = channel_list )
user_table.insert(0, "User_ID", user_list)

#Write to csv
user_table.to_csv("transformed_data/user_table_full.csv", index = False)

##########
#TABLE (2)
##########

#Time index; Count separate dates; Dates in string format for simplicity
date_list = attribs_convs["Conv_Date"].unique()
date_list = [np.datetime64(date) for date in date_list]
date_list.sort()
date_list = np.datetime_as_string(date_list) #sorted date list in string format

#Function that computes revenue given channel & date, taking into account IHC index;
def date_chan_revenue(Date, Channel):
	revenue = 0
	#Sub data frame of interest
	subdf = attribs_convs[(attribs_convs["Conv_Date"] == Date) & 
	(attribs_convs["Channel"] == Channel)]

	if not(subdf.empty):
		revenue = (subdf["IHC_Conv"]*subdf["Revenue"]).sum()
	
	return(revenue)

#As before, matrix to speed up computations
m = len(date_list)
p = len(channel_list)

ts_data = np.zeros((m, p)) #NP matrix to be filled with multivariate TS values

#Fill matrix
for i in range(0, m):
	#print("Inserting date ", i+1, "/", m) #To roughly evaluate execution time
	for j in range(0, p):
		ts_data[i][j] = date_chan_revenue(date_list[i], channel_list[j])

#Convert matrix to dataframe
ts_table = pd.DataFrame(ts_data, columns = channel_list )
ts_table.insert(0, "Conv_Date", date_list) #add date columm

#Write to csv
ts_table.to_csv("transformed_data/time_channels_table.csv", index = False)