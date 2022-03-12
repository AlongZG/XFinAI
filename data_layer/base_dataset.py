from torch.utils.data import Dataset


# Define Dataset BaseClass
class FuturesDataset(Dataset):

    def __init__(self, data, label, seq_length, features_list):
        self.data = data
        self.label = label
        self.features_list = features_list
        self.df_label = self.data[self.label]
        self.df_feature = self.data[self.features_list]
        self.seq_length = seq_length

        self.data_set = self.create_xy_pairs()
        self.data_length = len(self.data_set)

    def create_xy_pairs(self):
        pairs = []
        for idx in range(self.data.shape[0] - self.seq_length):
            x = self.df_feature[idx:idx + self.seq_length].values
            y = self.df_label[idx + self.seq_length:idx + self.seq_length + 1].values
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data_set[idx]
