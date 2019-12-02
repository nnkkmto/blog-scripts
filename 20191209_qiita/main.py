import pprint

import numpy as np
import pandas as pd

data = {}
data['userid'] = np.random.randint(0, 3, (21,))
data['itemid'] = np.random.randint(0, 10, (21,))
data['timestamp'] = np.array(['20190101','20190102','20190103','20190104','20190105','20190106','20190107','20190108','20190109',
                       '20190110','20190111','20190112','20190113','20190114','20190115','20190116','20190117','20190118',
                       '20190119','20190120','20190121'])
df = pd.DataFrame(data=data)
df['timestamp'] = pd.to_datetime(df['timestamp'])


data_cate = {}
data_cate['itemid'] = np.array([0,1,2,3,4,5,6,7,8,9,10])
data_cate['categoryid'] = np.array([0,0,0,1,1,1,2,2,2,3,3])
df_cate = pd.DataFrame(data=data_cate)
df = pd.merge(df, df_cate, how='left', on='itemid')
df = df[['userid','itemid','categoryid','timestamp']]
df = df.sort_values(by=['userid','itemid']).reset_index(drop=True)
print('例示用')
print(df)


# ユーザーid, 時系列順に並び替え
df_seq = df.sort_values(by=['userid','timestamp'])
# ユーザー単位でリストとしてgroupby
df_seq = df_seq.groupby('userid').agg(list).reset_index(drop=False)

# 値の取り出し
print('各ユーザーが接触したitemid（時系列順）')
pprint.pprint(df_seq['itemid'].values.tolist())
print('各ユーザーが接触したcategoryid（時系列順）')
pprint.pprint(df_seq['categoryid'].values.tolist())

# ユーザー単位で最新のものを取るようにgroupby
df_cate = df.loc[df.groupby('userid')['timestamp'].idxmax()]

print(df_cate)
print('各ユーザーが接触した最新のitemid')
pprint.pprint(df_cate['itemid'].values.tolist())
print('各ユーザーが接触した最新のcategoryid')
pprint.pprint(df_cate['categoryid'].values.tolist())

import tensorflow as tf
inputs = []
inputs.append(tf.keras.preprocessing.sequence.pad_sequences(
    df_seq['itemid'].values.tolist(), padding='post', truncating='post', maxlen=10))
inputs.append(tf.keras.preprocessing.sequence.pad_sequences(
    df_seq['categoryid'].values.tolist(), padding='post', truncating='post', maxlen=10))


def create_list(df, user_index_col, sort_col, target_col, user_num):
    """
    :param user_index_col: ユーザーIDのカラム
    :param sort_col: sortに使う値の入っているカラム
    :param target_col: sortしたいカラム
    :param user_num: ユーザー数（エンコーダ等から取得してください）
    """
    inputs = [[] for _ in range(user_num)]
    for _, user_index, sort_value, target_value in df[[user_index_col, sort_col, target_col]].itertuples():
        inputs[user_index].append([target_value, sort_value])

    return inputs


itemid_inputs = create_list(df, user_index_col='userid', sort_col='timestamp', target_col='itemid', user_num=3)
categoryid_inputs = create_list(df, user_index_col='userid', sort_col='timestamp', target_col='categoryid', user_num=3)

print('itemid')
pprint.pprint(itemid_inputs)

print('categoryid')
pprint.pprint(categoryid_inputs)

def sort_list(inputs, is_descending):
    """
    :param is_descending: 降順かどうか
    """
    return [sorted(i_input, key=lambda i: i[1], reverse=is_descending) for i_input in inputs]


itemid_inputs = sort_list(itemid_inputs, is_descending=False)
categoryid_inputs = sort_list(categoryid_inputs, is_descending=False)

print('itemid')
pprint.pprint(itemid_inputs)

print('categoryid')
pprint.pprint(categoryid_inputs)

def create_sequential(inputs):
    # リストのうちtimestampのリストを削除
    return [[i[0] for i in i_input] for i_input in inputs]

print('各ユーザーが接触したitemid（時系列順）')
pprint.pprint(create_sequential(itemid_inputs))

print('各ユーザーが接触したcategoryid（時系列順）')
pprint.pprint(create_sequential(categoryid_inputs))

def create_category(inputs, n=-1):
    """
    :param n: 時系列順のリストのうち、何番目のものを残すか
    """
    # リストのうちtimestampのリストを削除
    # 時系列順の系列データのうち、n番目のもののみを残す
    return [[i[0] for i in i_input][n] for i_input in inputs]

print('各ユーザーが接触した最新のitemid')
pprint.pprint(create_category(itemid_inputs, -1))

print('各ユーザーが接触した最新のcategoryid')
pprint.pprint(create_category(categoryid_inputs, -1))


def create_features(
        df, user_index_col, sort_col, target_col, user_num, is_descending, is_sequence, n=-1):
    """
    :param user_index_col: ユーザーIDのカラム
    :param sort_col: sortに使う値の入っているカラム
    :param target_col: sortしたいカラム
    :param user_num: ユーザー数（エンコーダ等から取得してください）
    :param is_descending: 降順かどうか
    :param is_sequence: シーケンシャルかどうか
    :param n: 時系列順のリストのうち、何番目のものを残すか（カテゴリーのみ）
    """
    # リストの作成
    inputs = [[] for _ in range(user_num)]
    for _, user_index, sort_value, target_value in df[[user_index_col, sort_col, target_col]].itertuples():
        inputs[user_index].append([target_value, sort_value])

    # リストのソート
    inputs = [sorted(i_input, key=lambda i: i[1], reverse=is_descending) for i_input in inputs]

    if is_sequence:
        return [[i[0] for i in i_input] for i_input in inputs]
    else:
        return [[i[0] for i in i_input][n] for i_input in inputs]

print(create_features(df, user_index_col='userid', sort_col='timestamp', target_col='itemid', user_num=3, is_descending=False, is_sequence=True))
print(create_features(df, user_index_col='userid', sort_col='timestamp', target_col='itemid', user_num=3, is_descending=False, is_sequence=False))

# 例示用
df1 = df[:7]
df2 = df[7:14]
df3 = df[14:21]

print(df1)
print(df2)
print(df3)

df_dict = {}
df_dict['df1'] = df1
df_dict['df2'] = df2
df_dict['df3'] = df3

pprint.pprint(df_dict)

def create_features_by_datasets(
        df_dict, user_index_col, sort_col, target_col, user_num, is_descending, is_sequence, n=-1):
    inputs = [[] for _ in range(user_num)]

    # データセットの分割単位ごとに対して処理
    for df in df_dict.values():
        for _, user_index, sort_value, target_value in df[[user_index_col, sort_col, target_col]].itertuples():
            inputs[user_index].append([target_value, sort_value])

    inputs = [sorted(i_input, key=lambda i: i[1], reverse=is_descending) for i_input in inputs]

    if is_sequence:
        return [[i[0] for i in i_input] for i_input in inputs]
    else:
        return [[i[0] for i in i_input][n] for i_input in inputs]

pprint.pprint(create_features_by_datasets(df_dict, user_index_col='userid', sort_col='timestamp', target_col='itemid', user_num=3, is_descending=False, is_sequence=True))
pprint.pprint(create_features_by_datasets(df_dict, user_index_col='userid', sort_col='timestamp', target_col='itemid', user_num=3, is_descending=False, is_sequence=False))


data = {}
data['userid'] = np.random.randint(0, 3, (21,))
data['itemid'] = np.random.randint(0, 10, (21,))
data['score'] = np.random.rand(21)

df = pd.DataFrame(data=data)


data_cate = {}
data_cate['itemid'] = np.array([0,1,2,3,4,5,6,7,8,9,10])
data_cate['categoryid'] = np.array([0,0,0,1,1,1,2,2,2,3,3])
df_cate = pd.DataFrame(data=data_cate)
df = pd.merge(df, df_cate, how='left', on='itemid')
df = df[['userid','itemid','categoryid','score']]
df = df.sort_values(by=['userid','itemid']).reset_index(drop=True)
print('例示用')
print(df)

print('スコア順（itemid）')
pprint.pprint(create_features(df, user_index_col='userid', sort_col='score', target_col='itemid', user_num=3, is_descending=True, is_sequence=True))
print('スコア最大（itemid）')
pprint.pprint(create_features(df, user_index_col='userid', sort_col='score', target_col='itemid', user_num=3, is_descending=True, is_sequence=False, n=0))