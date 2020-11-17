import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


data = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-08-07_database_tf.csv")
rawdata = data.loc[:,["date", "x_world", "x_cn", "bci_jp", "fc_x_de"]]

def gen_dataset(rawdata, target_variable):
	"generate a raw dataset to feed into the model, outputs np array of dimensions `n_obs x n_features + 1`"
	""" Intermediate step to generate a raw dataset the model will accept
	Input should be a pandas dataframe of of (n observations) x (m features + 1 target column). Non-numeric columns will be dropped, missing values replaced by 0.
	
	:rawdata: pandas DataFrame: n x m+1 dataframe
	:return: numpy array: n x m+1 array
	"""
	
	rawdata = rawdata.fillna(0.0) # fill nas with 0's
	# converting all columns to float64 if numeric
	for col in rawdata.columns:
		if is_numeric_dtype(rawdata[col]):
			rawdata[col] = rawdata[col].astype("float64")
	rawdata = rawdata.loc[:, [x == "float64" for x in rawdata.dtypes]] # only keep numeric columns
	variables = list(rawdata.columns[rawdata.columns != target_variable]) # features, excluding target variable
	
	data_dict = {}
	for variable in variables:
		data_dict[variable] = np.array(rawdata.loc[:,variable])
		data_dict[variable] = data_dict[variable].reshape((len(data_dict[variable]), 1))
	target = np.array(rawdata.loc[:,target_variable])
	target = target.reshape((len(target), 1))
	dataset = np.hstack(([data_dict[k] for k in data_dict] + [target]))
	return dataset