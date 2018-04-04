import torch.utils.data as data
import torch
from collections import defaultdict
import random

# TODO (anguelos) verify that this should not be a DataLoader
class SiameseDs(data.Dataset):
    def create_triplets(self, dataset, triplets_per_sample):
        self.triplet_idx_list = []

        per_class = defaultdict(list)
        self.classid_list = []

        rnd_sel = lambda x: x[random.randint(0, len(x) - 1)]
        for n, (img, class_id) in enumerate(dataset):
            per_class[class_id].append(n)
            self.classid_list.append(class_id)
        per_class_list = per_class.items()
        for _ in range(triplets_per_sample):
            for ancor_id in range(len(self.classid_list)):
                ancor_class = self.classid_list[ancor_id]
                near_id = rnd_sel(per_class[ancor_class])
                while near_id == ancor_id:
                    near_id = rnd_sel(per_class[ancor_class])
                far_class = rnd_sel(per_class_list)[0]
                while far_class == ancor_class:
                    far_class = rnd_sel(per_class_list)[0]
                far_id = rnd_sel(per_class[far_class])
                self.triplet_idx_list.append((ancor_id, near_id, far_id))
        return self.triplet_idx_list

    def __init__(self, dataset, triplets_per_sample=1, triplet=True, same_pair_prob=1.):
        self.triplet = triplet
        # if same_pair_prob
        self.same_pair_prob = same_pair_prob
        self.dataset = dataset
        self.create_triplets(dataset, triplets_per_sample)

    def __len__(self):
        return len(self.triplet_idx_list)

    def __getitem__(self, item):
        if self.triplet:
            ancor_id, near_id, far_id = self.triplet_idx_list[item]
            return self.dataset[ancor_id][0], self.dataset[near_id][0], self.dataset[far_id][0]
        else:
            ancor_id, near_id, far_id = self.triplet_idx_list[item]
            y = torch.rand(1)[0] < self.same_pair_prob
            if y:
                return self.dataset[ancor_id][0], self.dataset[near_id][0], torch.ones(1)
            else:
                return self.dataset[ancor_id][0], self.dataset[far_id][0], torch.zeros(1)
