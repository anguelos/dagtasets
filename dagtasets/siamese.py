import torch.utils.data as data
import torch
from collections import defaultdict
import random
from torch._six import int_classes as _int_classes
import numpy as np

# TODO (anguelos) verify that this should not be a DataLoader
class SiameseDs(data.Dataset):
    """Makes a siamese dataset out of a classification dataset.

    This class wraps an existing classification dataset and presents it as a
    siamese dataset suited either for triplet loss or for contrastive loss.

    :param triplets_per_sample: The triplets are constructed
    :param triplet: If true
    """
    def __init__(self, dataset, triplets_per_sample=1, triplet=True, same_pair_prob=1.):
        self.triplet = triplet
        # if same_pair_prob
        self.same_pair_prob = same_pair_prob
        self.dataset = dataset
        self.create_triplets(dataset, triplets_per_sample)

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

    def __len__(self):
        return len(self.triplet_idx_list)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
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


class BatchMiningSampler(torch.utils.data.sampler.BatchSampler):
    """Dataset batch sampler.

    :param data_source:
    :param class_per_batch:
    :param samples_per_class:
    :param drop_last:
    """
    """Dataset wrapper aimed at batch hard-negative mining.

    :param dataset: A dataset providing hashable targets, ideally integers.
    :param class_per_batch: How many classes will be represented in each batch.
    :param samples_per_class: How many samples for each class will exist in the minibatch.
    :param drop_last: If True, whatever samples
    """
    def __init__(self,data_source,class_per_batch=5,samples_per_class=5,drop_last=False):
        if not isinstance(data_source, torch.utils.data.Dataset):
            raise ValueError("data_source should be an instance of "
                             "torch.utils.data.Dataset, but got sampler={}"
                             .format(data_source))
        if not isinstance(class_per_batch, _int_classes) or isinstance(class_per_batch, bool) or \
                class_per_batch <= 0:
            raise ValueError("class_per_batch should be a positive integeral value, "
                             "but got class_per_batch={}".format(class_per_batch))
        if not isinstance(samples_per_class, _int_classes) or isinstance(samples_per_class, bool) or \
                samples_per_class <= 0:
            raise ValueError("samples_per_class should be a positive integeral value, "
                             "but got samples_per_class={}".format(samples_per_class))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        sz=len(data_source)
        samples_by_class=defaultdict(lambda:list())
        if hasattr(data_source,"get_target"):
            for k in range(sz):
                label = data_source.get_target(k)
                samples_by_class[label].append(k)
        else:
            for k in range(sz):
                _,label = data_source[k]
                samples_by_class[label].append(k)

        self.samples_by_class={k:np.array(v) for k, v in samples_by_class.items()}
        self.classes=np.array(self.samples_by_class.keys())
        self.class_per_batch=class_per_batch
        self.samples_per_class=samples_per_class
        self.drop_last=drop_last
        if drop_last:
            self.batch_count = int(np.floor(float(self.class_per_batch) / self.classes.shape[0]))
        else:
            self.batch_count = int(np.ceil(float(self.class_per_batch) / self.classes.shape[0]))
        print("batch count",repr(self.batch_count),repr(self.classes.shape))
        for cl_id in self.samples_by_class.keys():
            while len(self.samples_by_class[cl_id]) < self.samples_per_class:
                self.samples_by_class[cl_id] = np.concatenate([self.samples_by_class[cl_id],self.samples_by_class[cl_id]],axis=0)

    def __iter__(self):
        shuffled_classes=self.classes.copy()
        np.random.shuffle(shuffled_classes)
        if self.batch_count*self.class_per_batch*self.samples_per_class > self.classes.shape[0]:
            if not self.drop_last:
                shuffled_classes=np.concatenate((shuffled_classes,shuffled_classes),axis=0)[:self.batch_count*self.class_per_batch]
            else:
                shuffled_classes = shuffled_classes[:self.batch_count * self.class_per_batch]
        class_idx=0
        for _ in range(self.batch_count):
            batch=[]
            for _ in range(self.class_per_batch):
                class_id=shuffled_classes[class_idx]
                samples=self.samples_by_class[class_id].copy()
                np.random.shuffle(samples)
                if samples.shape[0]>self.samples_per_class:
                    batch+=samples[:self.samples_per_class].tolist()
                else:
                    idx=np.arange(self.samples_per_class,dtype="int32")%samples.shape[0]
                    batch+=samples[idx].tolist()
                class_idx+=1
            yield batch

    def __len__(self):
        return self.samples_per_class*self.class_per_batch*self.batch_count
