import os
from collections import defaultdict

import numpy as np


class KnowledgeGraph:
    def __init__(self, path, dataset):
        super(KnowledgeGraph, self).__init__()
        assert dataset in ['WN18', 'WN18RR', 'FB15k', 'FB15k-237', 'YAGO3-10']

        train_path = os.path.join(path, dataset, 'train2id.txt')
        valid_path = os.path.join(path, dataset, 'valid2id.txt')
        test_path = os.path.join(path, dataset, 'test2id.txt')
        entity_path = os.path.join(path, dataset, 'entity2id.txt')
        relation_path = os.path.join(path, dataset, 'relation2id.txt')

        self.entity2id = {}
        self.relation2id = {}
        with open(entity_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                entity, id = line.strip().split('\t')
                self.entity2id[str(entity)] = int(id)
        with open(relation_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                relation, id = line.strip().split('\t')
                self.relation2id[str(relation)] = int(id)
        self.num_entity = len(self.entity2id)
        self.num_relation = 2 * len(self.relation2id)

        self.train_data = self.read(train_path)
        self.valid_data = self.read(valid_path)
        self.test_data = self.read(test_path)
        self.data = np.concatenate((self.train_data, self.valid_data, self.test_data), axis=0)

        self.hr_vocab = defaultdict(list)
        for triplet in self.data:
            self.hr_vocab[(triplet[0], triplet[1])].append(triplet[2])

    def read(self, file_path):
        with open(file_path, "r", encoding='UTF-8') as f:
            lines = f.readlines()
        triplets = np.zeros((2 * len(lines), 3), dtype=np.int64)
        for i, line in enumerate(lines):
            h, r, t = line.strip().split('\t')
            triplets[2 * i] = np.array([int(h), int(r), int(t)])
            triplets[2 * i + 1] = np.array([int(t), int(r) + self.num_relation // 2, int(h)])
        return triplets
