import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def rawdf_to_stockdfs(df):
    """
    Takes a raw pd DataFrame and creates a list of data frames for each stock. Also, drops stock_id, date_id, time_id, and row_id.
    """
    unique_stock_ids = df.stock_id.unique()
    list_df = []
    for id in unique_stock_ids:
        df_it = df[df['stock_id'] == id]
        df_it = df_it.drop(columns=['stock_id', 'date_id', 'time_id', 'row_id'], inplace=False)
        df_it = torch.tensor(df_it.to_numpy(), dtype=torch.float)
        list_df.append(df_it)
    return list_df

def data_to_sequences(data, seq_length):
    sequences = data.unfold(0, seq_length, 1)
    sequences = torch.transpose(sequences, 1, 2)
    return sequences

def batch_tensor(tensor, batch_size):
    """
    Splits a tensor into batches along the first dimension.

    Args:
    tensor (torch.Tensor): The input tensor to be batched.
    batch_size (int): The size of each batch.

    Returns:
    list of torch.Tensor: A list of the batches.
    """
    # Split the tensor into batches along the first dimension
    batches = torch.split(tensor, batch_size)
    return list(batches)

def df_list_to_series_tensor(data_list, sequence_lenght, shuffle=True):
    # Turning data into sequences
    tensor_list = []
    for i in range(len(data_list)):
        sequences = data_to_sequences(data_list[i], sequence_lenght)
        tensor_list.append(sequences)
    # Aggregating the sequences in one tensor
    collected_tensor = torch.cat(tensor_list, dim=0)

    if shuffle:
        # Shuffling the sequences
        shuffled_ind = torch.randperm(collected_tensor.size(0))
        collected_tensor = collected_tensor[shuffled_ind]
    return collected_tensor

def get_features(data):   
    # imbalance 
    data['imbalance'] = data.imbalance_size * data.imbalance_buy_sell_flag
    data['relative_imbalance'] = data.imbalance_size / data.matched_size
    data['competitive_imbalance'] = data.ask_size - data.bid_size
    data['spread'] = data.ask_price - data.bid_price
    data['competitive_price'] = data.ask_size*data.ask_price - data.bid_size*data.bid_price
    
    # time
    data.insert(0, 'dow', data["date_id"] % 5) # days of the week
    data.insert(1, 'minute', data["seconds_in_bucket"] // 60) # minutes
    data.insert(2, 'seconds', data["seconds_in_bucket"] % 60) # seconds

    # delta
    data['ref_far_delta'] = data.reference_price - data.far_price
    data['ref_near_delta'] = data.reference_price - data.near_price
    data['far_near_delta'] = data.far_price - data.near_price
    data['ref_wap_delta'] = data.reference_price - data.wap
    data['near_wap_delta'] = data.near_price - data.wap
    data['far_wap_delta'] = data.far_price - data.wap
    data['ref_bid_delta'] = data.reference_price - data.bid_price
    data['ref_ask_delta'] = data.reference_price - data.ask_price
    
    # ratio
    data['ref_wap_ratio'] = data.reference_price / data.wap
    data['near_wap_ratio'] = data.near_price / data.wap
    data['far_wap_ratio'] = data.far_price / data.wap
    
    # index
    data['index_imbalance'] = data.groupby('seconds_in_bucket', group_keys=False)['imbalance'].transform('mean')
    data['indexref_price'] = data.groupby('seconds_in_bucket', group_keys=False)['reference_price'].transform('mean')
    data['indexwap_price'] = data.groupby('seconds_in_bucket', group_keys=False)['wap'].transform('mean')
    data['indexibmalance_delta'] = data.imbalance - data.index_imbalance
    data['ref_indexref_delta'] = data.reference_price - data.indexref_price
    data['ref_indexwap_delta'] = data.reference_price - data.indexwap_price
    data['indexibmalance_ratio'] = data.imbalance / data.index_imbalance
    data['ref_indexref_ratio'] = data.reference_price / data.indexref_price
    data['ref_indexwap_ratio'] = data.reference_price / data.indexwap_price
    
    # ema
    data['ema5_imbalance'] = data.groupby('stock_id', group_keys=False)['imbalance'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_rel_imb'] = data.groupby('stock_id', group_keys=False)['relative_imbalance'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_com_imb'] = data.groupby('stock_id', group_keys=False)['competitive_imbalance'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_spread'] = data.groupby('stock_id', group_keys=False)['spread'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_comp_pri'] = data.groupby('stock_id', group_keys=False)['competitive_price'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_ref_pri'] = data.groupby('stock_id', group_keys=False)['reference_price'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_far_pri'] = data.groupby('stock_id', group_keys=False)['far_price'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_near_pri'] = data.groupby('stock_id', group_keys=False)['near_price'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_wap_pri'] = data.groupby('stock_id', group_keys=False)['wap'].transform(lambda x: x.ewm(span=5).mean())
    
    data['ema5_r_f_delta'] = data.groupby('stock_id', group_keys=False)['ref_far_delta'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_r_n_delta'] = data.groupby('stock_id', group_keys=False)['ref_near_delta'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_f_n_delta'] = data.groupby('stock_id', group_keys=False)['far_near_delta'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_r_w_delta'] = data.groupby('stock_id', group_keys=False)['ref_wap_delta'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_n_w_delta'] = data.groupby('stock_id', group_keys=False)['near_wap_delta'].transform(lambda x: x.ewm(span=5).mean())
    data['ema5_f_w_delta'] = data.groupby('stock_id', group_keys=False)['far_wap_delta'].transform(lambda x: x.ewm(span=5).mean())
    
    data['log_bid_price'] = np.log1p(data['bid_price'])
    data['log_ask_price'] = np.log1p(data['ask_price'])
    data['log_wap'] = np.log1p(data['wap'])
    data['log_reference_price'] = np.log1p(data['reference_price'])
 

    data['bid_ask_ratio'] = data['bid_price'] / data['ask_price']
    data['imbalance_squared'] = data['imbalance'] ** 2
    data['ref_wap_ratio'] = data['reference_price'] / data['wap']

    data['wap_ema'] = data.groupby('stock_id')['wap'].transform(lambda x: x.ewm(span=5).mean())
    data['wap_ema_delta'] = data['wap_ema'] - data['wap']
    data['bid_ask_spread'] = data['ask_price'] - data['bid_price']

    data["volume"] = data.eval("ask_size + bid_size")

    return data

def complete_data_preparation(sequence_length, batch_size, data_file_path = '../data/data.parquet.gzip'):
    """
    This function enables complete data preparation with different sequence lengths and batch sizes.
    This enables simple inclusion of these values in the hyperparameter optimization.
    """
    raw_df = pd.read_parquet(data_file_path)
    # In order to use ffill later
    raw_df.sort_values(by=['stock_id', 'time_id'], inplace=True)

    # Fill NA values with 0 when possible and using the last valid observation otherwise
    raw_df.fillna({
        'far_price': 0,
        'near_price': 0,
        'imbalance_size': raw_df['imbalance_size'].ffill(),
        'reference_price': raw_df['reference_price'].ffill(),
        'matched_size': raw_df['matched_size'].ffill(),
        'bid_price': raw_df['bid_price'].ffill(),
        'ask_price': raw_df['ask_price'].ffill(),
        'wap': raw_df['wap'].ffill()
    }, inplace=True)

    # Amplifying the Data with Features engineered by hand
    amp_df = get_features(raw_df)

    # Reordering the columns to ensure Targets is the last column
    new_order = [col for col in amp_df.columns if col != 'target'] + ['target']
    amp_df = amp_df[new_order]

    amp_df = amp_df.dropna()

    df_train = amp_df[amp_df['date_id'] < 100]
    df_train = df_train.sort_values(by=['stock_id', 'time_id'], inplace=False)

    df_validation = amp_df[amp_df['date_id'] > 99]
    df_validation = df_validation[df_validation['date_id'] < 120]
    df_validation = df_validation.sort_values(by=['stock_id', 'time_id'], inplace=False)

    df_test = amp_df[amp_df['date_id'] > 119]
    df_test = df_test[df_test['date_id'] < 140]
    df_test = df_test.sort_values(by=['stock_id', 'time_id'], inplace=False)

    # Scaling the Data
    # The first four will later be dropped and the target column will be scaled differently
    excluded_columns = ['stock_id', 'date_id', 'time_id', 'row_id', 'target']

    # Standardize only the columns that are not in the excluded list
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Scaling the Training Data
    df_train[df_train.columns.difference(excluded_columns)] = feature_scaler.fit_transform(df_train[df_train.columns.difference(excluded_columns)])
    df_train[['target']] = target_scaler.fit_transform(df_train[['target']])

    # Scaling the Validation Data
    df_validation[df_validation.columns.difference(excluded_columns)] = feature_scaler.transform(df_validation[df_validation.columns.difference(excluded_columns)])
    df_validation[['target']] = target_scaler.transform(df_validation[['target']])

    # Scaling the Test Data
    df_test[df_test.columns.difference(excluded_columns)] = feature_scaler.transform(df_test[df_test.columns.difference(excluded_columns)])
    df_test[['target']] = target_scaler.transform(df_test[['target']]) 
    
    dflist_train = rawdf_to_stockdfs(df_train)
    dflist_validation = rawdf_to_stockdfs(df_validation)
    dflist_test = rawdf_to_stockdfs(df_test)
    
    # Turning data into sequences
    collected_tensor_train = df_list_to_series_tensor(dflist_train, sequence_length, shuffle=True)
    collected_tensor_test = df_list_to_series_tensor(dflist_test, sequence_length, shuffle=True)
    collected_tensor_validation = df_list_to_series_tensor(dflist_validation, sequence_length, shuffle=True)

    # Turning the aggregated sequences into batches
    batches_train = batch_tensor(collected_tensor_train, batch_size)
    batches_validation = batch_tensor(collected_tensor_test, batch_size)
    batches_test = batch_tensor(collected_tensor_validation, batch_size)
    
    return batches_train, batches_validation, batches_test