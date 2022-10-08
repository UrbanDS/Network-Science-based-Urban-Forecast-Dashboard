import pandas as pd
import json
from glob import glob
from prophet import Prophet

def build_geo_node():
    age_ids = ["B01001e"+str(i) for i in range(1,50)]
    sum_ids = ["B00001e1","B00002e1"]
    race_ids = ["B02001e"+str(i) for i in range(1,10)]
    table_ids = ['B19013e1','B19001e1', 'B01002e1'] + age_ids + race_ids#+ sum_ids
    cbg_field_desc = pd.read_csv('safegraph_open_census_data_2020/metadata/cbg_field_descriptions.csv')
    cbg_field_desc[cbg_field_desc.table_id.isin(table_ids)]
    cbg_b19 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b19.csv', dtype={'census_block_group': str})
    cbg_b01 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b01.csv', dtype={'census_block_group': str})
    cbg_b02 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b02.csv', dtype={'census_block_group': str})
    # cbg_b00 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b00.csv', dtype={'census_block_group': str})
    cbg_data = pd.merge(cbg_b01, cbg_b19, on=['census_block_group'], how='inner') #, cbg_b00,on=['census_block_group'])
    cbg_data = pd.merge(cbg_data, cbg_b02, on=['census_block_group'], how='inner')
    # criterion = cbg_data['census_block_group'].map(lambda x: x.startswith('48'))
    # cbg_data = cbg_data[criterion]
    cbg_data = cbg_data[['census_block_group'] + table_ids]
    # cbg_data.dropna().head()
    return cbg_data


def prepare_data():
    patterns_df = pd.read_csv("patterns2021.csv")
    patterns_feb = patterns_df[patterns_df['date_range_start'] == '2021-02-01T00:00:00-06:00']
    patterns_feb = patterns_feb.dropna(subset=['distance_from_home'])
    poi_df = pd.read_csv("places2021.csv")
    poi_df = poi_df[poi_df['placekey'].isin(patterns_feb['placekey'])]
    patterns_feb = patterns_feb[patterns_feb['placekey'].isin(poi_df['placekey'])]
    poi_df = poi_df.join(patterns_feb.set_index('placekey'), on='placekey', how='inner', lsuffix='_poi', rsuffix='_patterns')

    #merge the cbg data with the poi data
    cbg_data = build_geo_node()
    cbg_data = cbg_data.rename(columns={"visitor_daytime_cbgs_y": "census_block_group"})
    poi_df['visitor_daytime_cbgs'] = [json.loads(cbg_json) for cbg_json in poi_df.visitor_daytime_cbgs]

    # extract each key:value inside each visitor_home_cbg dict (2 nested loops)
    all_sgpid_cbg_data = []  # each cbg data point will be one element in this list
    for index, row in poi_df.iterrows():
        this_sgpid_cbg_data = [
            {'placekey': row['placekey'], 'visitor_daytime_cbgs': key, 'visitor_count': value} for
            key, value in row['visitor_daytime_cbgs'].items()]

        # concat the lists
        all_sgpid_cbg_data = all_sgpid_cbg_data + this_sgpid_cbg_data
    # note: visitor_cbg_data_df has 3 columns: safegraph_place_id, visitor_count, visitor_daytime_cbgs
    visitor_cbg_data_df = pd.DataFrame(all_sgpid_cbg_data)
    #len(visitor_cbg_data_df['visitor_daytime_cbgs'].unique())
    cbg_data = cbg_data[cbg_data['census_block_group'].isin(visitor_cbg_data_df['visitor_daytime_cbgs'])] #key is census_block_group
    cbg_data = cbg_data.fillna(0)
    # ignore cbg that not in the cbg_data
    visitor_cbg_data_df = visitor_cbg_data_df[visitor_cbg_data_df['visitor_daytime_cbgs'].isin(cbg_data['census_block_group'])]
    list_new_cbg_columns = []
    for cbg in cbg_data['census_block_group'].unique():
        for c in cbg_data.columns:
            list_new_cbg_columns.append(cbg + '_' + c)
    temp_df = pd.DataFrame(columns=list_new_cbg_columns)
    poi_df = pd.concat([poi_df, temp_df], axis=1).fillna(0)
    cbg_data = cbg_data.set_index('census_block_group')
    for index, row in poi_df.iterrows():
        for key, value in row['visitor_daytime_cbgs'].items():
            try:
                if poi_df.loc[index, key + '_census_block_group'] == 0:
                    poi_df.loc[index, key + '_census_block_group'] = value
                else:
                    print('error', key + '_census_block_group')
                for c in cbg_data.columns:
                    if c == 'census_block_group':
                        continue
                    else:
                        poi_df.loc[index, key + '_' + c] = cbg_data.loc[key, c]
            except KeyError:
                continue

    census_x = poi_df[list_new_cbg_columns].values




    # poi_df = poi_df.merge(visitor_cbg_data_df, on='placekey', how='inner')
    # poi_df = poi_df.merge(cbg_data, on='visitor_daytime_cbgs_y', how='inner')

    # get the edges
    location_poi = poi_df[['latitude', 'longitude']]
    nbrs = NearestNeighbors(n_neighbors=5).fit(location_poi)
    distances, indices = nbrs.kneighbors(location_poi)
    index = torch.LongTensor(indices)
    length = index.size(1)
    source_index = torch.arange(0, index.size(0)).repeat_interleave(length)
    target_index = index.flatten()
    edge_index = torch.stack([source_index, target_index])

    # get the visit counts
    visits_by_day = np.array([json.loads(cbg_json) for cbg_json in poi_df.visits_by_day])
    real_y = torch.FloatTensor(visits_by_day[:, 10:20])

    # get the node features
    list_numerical_features = ['latitude', 'longitude']
    list_text_features = ['top_category']
    text_vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
    list_text = poi_df[list_text_features].astype(str).agg(' '.join, axis=1).to_list()
    text_vectorizer.fit(list_text)
    print("text emb dimension:")
    print(len(text_vectorizer.vocabulary_))
    emb_text = text_vectorizer.transform(list_text).todense()

    x = np.concatenate([poi_df[list_numerical_features], emb_text, census_x], axis=1)
    x = torch.FloatTensor(x)

    return x, edge_index, real_y


def baseline_linear_regression():
    x, edge_index, real_y = prepare_data()
    # linear regression model that regress x on real_y
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    mse_list = []
    for i in range(0, real_y.size(1)):
        regr.fit(x, real_y[:,i])
        predicted_yi = regr.predict(x)
        mse = mean_squared_error(real_y[:,i], predicted_yi)
        mse_list.append(mse)
    print(regr.score(x, real_y))


def baseline_mean(real_y):
    real_y_mean = torch.mean(real_y)
    F.mse_loss(real_y_mean, real_y)


if __name__ == '__main__':
    files = glob("annual2022/*")
    list_df = []
    for file in files:
        df = pd.read_csv(file)
        file_date = file.replace(".csv", "").replace("annual\\","")
        df["date"] = file_date
        list_df.append(df)
    df_all = pd.concat(list_df)
    grouped = df_all.groupby("date")
    # df_all.to_csv("1922poi_predicted.csv")
    # df1 = df_all[df_all.groupby(['placekey'])['placekey'].transform('count') > 484]
    # fil = df_all.groupby('placekey').filter(lambda x: len(x) > 484)
    loc_index = df_all.placekey.unique()
    list_df = []
    for i in loc_index:
        df = df_all[df_all.placekey == i]
        df = df.dropna(subset=['visit_count'])
        df['ds'] = df['date']
        df['y'] = df['visit_count']
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=100)
        future.tail()
        list_df.append(df)


