from torch.utils.data import Dataset


# Define Dataset BaseClass
class FuturesDatasetRecurrent(Dataset):

    def __init__(self, data, label, seq_length):
        self.data = data
        self.label = label

        self.df_label = self.data[self.label]
        self.df_feature = self.data[self.features_list]
        self.seq_length = seq_length

        self.data_x = []
        self.data_y = []
        self.data_set = self.create_xy_pairs()
        self.data_length = len(self.data_set)

    @property
    def features_list(self):
        features_list = list(self.data.columns)
        features_list.remove(self.label)
        return features_list

    def create_xy_pairs(self):
        pairs = []

        for idx in range(self.data.shape[0] - self.seq_length):
            x = self.df_feature[idx:idx + self.seq_length].values
            y = self.df_label[idx + self.seq_length - 1:idx + self.seq_length].values
            pairs.append((x, y))
            self.data_x.append(x)
            self.data_y.append(y)
        return pairs

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data_set[idx]


class FuturesDatasetTable(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

        self.df_label = self.data[self.label]
        self.df_feature = self.data[self.features_list]

        self.data_set = self.create_xy_pairs()
        self.data_length = len(self.data_set)

    @property
    def features_list(self):
        features_list = list(self.data.columns)
        features_list.remove(self.label)
        return features_list

    def create_xy_pairs(self):
        pairs = [(self.df_feature.iloc[idx].values, self.df_label.iloc[idx]) for idx in range(self.data.shape[0])]
        return pairs

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data_set[idx]
