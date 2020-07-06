import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Examples of general statistics

#Read previously transformed data
user_table = pd.read_csv("transformed_data/user_table_full.csv")
time_data = pd.read_csv("transformed_data/time_channels_table.csv")

#User Analysis

#User list
user_list = list(user_table["User_ID"])
n_users = len(user_list) #number of customers

#Keep only values
user_table = user_table.iloc[:,1:]

#test reduced ds
#user_table = user_table.iloc[1:5000, :]

#Analyze revenue per channel
avg_revs_chan = user_table.mean(axis = 0) #average revenue per chan
total_revs_chan = user_table.sum(axis = 0) #total revenue per chan
total_revenue = total_revs_chan.sum() #total revenue

#Extract most relevant channels by total revenue; group secondary
frac_revs_chan = total_revs_chan/total_revs_chan.sum() #fraction of revenue

#Primary: fraction revenue >5%
primary = frac_revs_chan[frac_revs_chan > 0.05]

#Lists of primary and secondary channels
channels = list(user_table.iloc[:,1:]) #all channels
primary_list = list(primary.index) #['A', 'B', 'E', 'G', 'H', 'I']
secondary_list = list(set(channels) - set(primary_list))

#Add total revenue for secondary channels; ~1.26e+06
total_revs_chan["Secondary"] = total_revs_chan[secondary_list].sum( axis = 0)
secondary_revenue_fraction = np.round((total_revs_chan["Secondary"])/total_revenue, 3) #~9%
primary_revenue_fraction = 1 - secondary_revenue_fraction
total_revs_chan = total_revs_chan[primary_list + ["Secondary"]]

#Print  some general statistics
print("Total revenue: ", np.round(total_revenue, 2))
print("Number of customers: ", n_users)
print("Primary channels", primary_list)
print("Fraction of revenue due to primary channels: ", np.round(primary_revenue_fraction * 100, 2), "%" )
print("Average spending per user in primary channels:", np.round(avg_revs_chan[primary_list],2), sep ="\n")
# --> High avg spending in A and G

#Plot
total_revs_chan.plot.pie()
plt.legend()
plt.ylabel("Revenue")
plt.title("Revenue share per channel")
plt.show()


#Channel Correlation
corr_matrix = np.round(user_table.corr(), 3)
sn.heatmap(corr_matrix) #Correlation heatmap graphic
plt.title("Channel Correlation (based on user spending)")
plt.show()
# --> Channels appear to be quite uncorrelated; little correlation between A and B (~21%)


'''
#Power Bi data generation
#Correlation matrix
corr_matrix.to_csv("powerbi_data/chan_correlation.csv")
'''

#Time Analysis

#Plot spending over time (through primary channels)
time_data[primary_list].plot()
plt.title("Revenue over time through main channels")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.show()

#Relevant peaks become visible
#Sum spending over all channels
all_chans = time_data[channels].sum(axis = 1)
time_data.insert(23, "Total", all_chans) #add total spending

#Plot total spending over time
plt.plot(time_data[ "Total"])
plt.title("Total Revenue over time")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.show()


