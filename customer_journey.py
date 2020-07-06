import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sn 

#Read data
conversions = pd.read_csv("data/table_A_conversions.txt")
attributions = pd.read_csv("data/table_B_attribution.txt")

#Merge tables - Perform outer join (do not eliminate nans)
raw_data = pd.merge(attributions, conversions, how = "outer", on = "Conv_ID")

#List of user IDs
user_list = list(raw_data["User_ID"].unique())

#Channel_list
channel_list = list(raw_data["Channel"].sort_values().unique())


#Class for user-specific statistics; takes as inout User_ID
class Customer_Journey:

	def __init__(self, User_ID):
		self.user = User_ID
		#Determine all conversions of user; sort by date
		self.convs = conversions[conversions["User_ID"] == ran_u].sort_values(by = ["Conv_Date"])
		#Attributions of user's conversions
		self.attribs = attributions[attributions["Conv_ID"].isin(self.convs["Conv_ID"])]

		#Main parameters of interest
		self.n_convs = len(self.convs.index) #numbers of conversions
		self.is_return = self.n_convs > 1 #is he/she a return customer?
		self.tot_spending = self.convs["Revenue"].sum() #Total revenue
		self.mean_spending = self.convs["Revenue"].mean() #Mean spending
		self.std_dev = self.convs["Revenue"].std() #Standard dev

	#Routine to compute user's attributions data
	def compute_journey_attributions(self):
		#Empty dataframe to store transformed attribution data
		journey_attrib_df = pd.DataFrame(columns = ["Conv_Date", "Channel", "Conv_ID"
 										"IHC_Conv", "Fraction_Revenue"])

		#Compute dataframe 
		for c_id in self.convs["Conv_ID"].values:
			c_date = self.convs[self.convs["Conv_ID"] == c_id].get("Conv_Date").values[0] #conversion date
			c_rev = self.convs[self.convs["Conv_ID"] == c_id].get("Revenue").values[0] #conversion revenue
			c_data = self.attribs[self.attribs["Conv_ID"] == c_id][["Channel", "IHC_Conv"]] #conversion data; add attributions 
			c_data["Conv_Date"] = c_date #Add date to conversion data
			c_data["Fraction_Revenue"] = c_rev * c_data["IHC_Conv"] #Fraction of revenue
			c_data["Conv_ID"] = c_id #Conversion ID as reference
			journey_attrib_df = pd.concat([journey_attrib_df, c_data])

		return journey_attrib_df

	#Print summary statistics
	def statistics_summary(self):
		#Channels general statistics
		self.channel_relevance = self.compute_journey_attributions().groupby("Channel")["Fraction_Revenue"].sum()
		self.visited_channels = list(self.channel_relevance.index)
		self.main_channel = self.visited_channels[self.channel_relevance.argmax()]

		print("User ID: ", self.user)
		print("Visited channels: ", self.visited_channels)
		print("Most influential channel: ", self.main_channel)
		print("Conversions count: ", self.n_convs)
		print("Total revenue:", self.tot_spending)
		print("Average revenue per conversion: ", self.mean_spending)
		print("Standard Deviation: ", self.std_dev)

	#Visualize channel influence over user by pie chart
	def channel_relevance_visualization(self):
		self.channel_relevance = self.compute_journey_attributions().groupby("Channel")["Fraction_Revenue"].sum()
		self.channel_relevance.plot.pie()
		plt.title("Channel Relevance for User")
		plt.ylabel("Share of spending per channel")
		plt.legend(loc = "upper right")
		plt.show()

	#Detailed plot of attributions
	def attribution_visualization(self):
		#Plot Journey attributions over time
		data = self.compute_journey_attributions()
		sn.barplot(x = "Conv_Date", y = "Fraction_Revenue", hue = "Channel", data = data)
		plt.title("(Single) Customer revenue per channels over time")
		plt.xlabel("Date")
		plt.ylabel("Spending")
		plt.show()

	
	#Write Journey Attributions in csv table format
	def write_journey_table(self):
		#Compute table
		data = self.compute_journey_attributions()[["Conv_Date", "Channel", "Fraction_Revenue"]]
		#Write
		data.to_csv("result_data/journey_attributions.csv", index = False)

#An example of usage
#Reproducibility
np.random.seed(35)

#Random User
ran_u = user_list[np.random.randint(len(user_list))]

journey = Customer_Journey(ran_u)

#Print statistics
journey.statistics_summary()

#Plot channel relevance
journey.channel_relevance_visualization()

#Plot journey
journey.attribution_visualization()

#Write journey data to csv format
journey.write_journey_table()