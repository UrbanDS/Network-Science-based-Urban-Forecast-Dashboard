import os.path as osp
import argparse
import pdb
import torch
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer


def calculate_r_square(y, y_pred):
    return 1 - np.sum(np.square(y - y_pred)) / max(np.sum(np.square(y - np.mean(y))), 1e-16)


def build_geo_node():
    age_ids = ["B01001e"+str(i) for i in range(1,50)]
    sum_ids = ["B00001e1","B00002e1"]
    table_ids = ['B19013e1','B19001e1', 'B01002e1']  # + age_ids + sum_ids
    cbg_field_desc = pd.read_csv('safegraph_open_census_data_2020/metadata/cbg_field_descriptions.csv')
    cbg_field_desc[cbg_field_desc.table_id.isin(table_ids)]
    cbg_b19 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b19.csv', dtype={'census_block_group': str})
    cbg_b01 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b01.csv', dtype={'census_block_group': str})
    # cbg_b00 = pd.read_csv('safegraph_open_census_data_2020/data/cbg_b00.csv', dtype={'census_block_group': str})
    cbg_data = pd.merge(cbg_b01, cbg_b19, on=['census_block_group']) #, cbg_b00,on=['census_block_group'])
    # criterion = cbg_data['census_block_group'].map(lambda x: x.startswith('48'))
    # cbg_data = cbg_data[criterion]
    cbg_data = cbg_data[['census_block_group'] + table_ids]
    # cbg_data.dropna().head()
    return cbg_data
    # cbg_geos = gpd.read_file(folder_path+'/geometry/cbg.geojson')
    #cbg_geos = cbg_geos[cbg_geos['State']=='PA' & cbg_geos['County']=='Allegheny']
    # cbg = cbg_geos.rename(columns={'CensusBlockGroup':'census_block_group'})[['census_block_group', 'geometry']]
    # criterion = cbg['census_block_group'].map(lambda x: x.startswith('42003'))
    # cbg = cbg[criterion & cbg.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    # cbg_data = cbg.merge(cbg_data, on='census_block_group',how='inner')
    # d = {'emb': [list(x) for x in zip(cbg_data.geometry.centroid.y, cbg_data.geometry.centroid.x)],
    #      'node_emb': np.concatenate([np.array(cbg_data[table_ids].fillna(cbg_data[table_ids].mean())),
    #                                  np.zeros([len(cbg_data), 400 - len(table_ids)])], axis=1).tolist(),
    #      'citation': [1] * len(cbg_data)}
    # node = pd.DataFrame(data=d, index=cbg_data['census_block_group'].to_list())
    # return node, cbg


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
    real_y = torch.FloatTensor(visits_by_day[:, 7:14])

    # get the node features
    list_numerical_features = ['latitude', 'longitude']
    list_text_features = ['top_category']
    text_vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
    list_text = poi_df[list_text_features].astype(str).agg(' '.join, axis=1).to_list()
    text_vectorizer.fit(list_text)
    print("text emb dimension:")
    print(len(text_vectorizer.vocabulary_))
    emb_text = text_vectorizer.transform(list_text).todense()

    x = np.concatenate([poi_df[list_numerical_features], torch.FloatTensor(visits_by_day[:, 0:7]), emb_text, census_x], axis=1)
    x = torch.FloatTensor(x)

    return x, edge_index, real_y


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_days=10):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, num_days, normalize=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='logs',
                        help='experiment name')
    args = parser.parse_args()
    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    # transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    # dataset = Planetoid(path, dataset, transform=transform)
    # data = dataset[0]
    x, edge_index, real_y = prepare_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(input_dim = x.size(1)).to(device)
    # data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x = x.to(device)
    edge_index = edge_index.to(device)
    real_y = real_y.to(device)
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.mse_loss(log_logits, real_y)
        # loss = kl_loss(torch.log(log_logits), real_y)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pdb.set_trace()
        log_logits = model(x, edge_index)
        loss = F.mse_loss(log_logits, real_y)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))
        # eb = model.conv1(x, edge_index)
        # with open("v1.npy", 'wb') as f: np.save(f, eb.detach().cpu().numpy())

    # explainer = GNNExplainer(model, epochs=200, return_type='log_prob')
    # node_idx = 10
    # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index,
    #                                                    edge_weight=edge_weight)
    # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    # plt.show()