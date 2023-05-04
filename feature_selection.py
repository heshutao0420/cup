import argparse
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-tx","--train_x", required = True, help="training x data full path including file name")
parser.add_argument("-ty","--train_y", required = True, help="training y data full path including file name")
parser.add_argument("-Ox","--output_x",required=True, help="output file full path including  file name")
parser.add_argument("-Oy","--output_y",required=True, help="output file full path including  file name")
parser.add_argument("-M","--method", required=True, help="Machine learning method")
args = parser.parse_args()

input_x = args.train_x
input_y = args.train_y
outnput_x = args.output_x
outnput_y = args.output_y

filename = input_x
train_selected_data = pd.read_csv(filename, index_col=0)
train_selected_y = pd.read_csv(input_y,index_col=0)
train_selected_y = train_selected_y["x"]

train_selected_data_for_feature_selection = train_selected_data
train_selected_data_for_feature_selection.index = train_selected_y.values.ravel()
train_selected_data_for_feature_selection_normal = train_selected_data_for_feature_selection.loc["normal"]

total_selcted_features = []
total_cancer_types_list = train_selected_y.drop_duplicates().values.ravel()
for num in range(len(total_cancer_types_list)-1):
    selected_cancer_data = train_selected_data_for_feature_selection.loc[total_cancer_types_list[num]]
    other_cancer_types = [n for n in total_cancer_types_list if n not in [total_cancer_types_list[num],"normal"]]
    other_cancer_data = train_selected_data_for_feature_selection.loc[other_cancer_types]
    meregd_data_df1 = pd.concat([selected_cancer_data,other_cancer_data], axis = 0)
    f1 = lambda x:stats.ranksums(x[0:selected_cancer_data.shape[0]],x[selected_cancer_data.shape[0]:meregd_data_df1.shape[0]]).pvalue
    t1 = meregd_data_df1.apply(f1)
    single_selected_features_1 = meregd_data_df1.columns[t1 <= 0.01]
    meregd_data_df2 = pd.concat([selected_cancer_data,train_selected_data_for_feature_selection_normal], axis = 0)
    f2 = lambda x:stats.ranksums(x[0:selected_cancer_data.shape[0]],x[selected_cancer_data.shape[0]:meregd_data_df2.shape[0]]).pvalue
    t2 = meregd_data_df2.apply(f1)
    single_selected_features_2 = meregd_data_df2.columns[t2 <= 0.01]
    single_selected_features = list(set(single_selected_features_1).intersection(set(single_selected_features_2)))
    total_selcted_features.append(single_selected_features)

final_selected_features = list(set([item for sublist in total_selcted_features for item in sublist]))
train_selected_data = train_selected_data[final_selected_features]
train_selected_data = train_selected_data.iloc[train_selected_data.index != "normal",]
train_selected_y = train_selected_y[train_selected_y != "normal"]
train_selected_data

train_selected_data.to_csv(outnput_x)
train_selected_y.to_csv(outnput_y)

