import numpy as np
import torch
from time import time
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from utils import Test_Sparse_Mat
from torch_sparse import spspmm
from heapq import nlargest
import random

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, dataset='gowalla'):
        # train or test
        path = f"./data/{dataset}"
        self.dataset_name = dataset
        print(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.split = False

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"The interactions of {self.dataset_name} is {self.trainDataSize + self.testDataSize}")
        print(f"User: {self.n_user}, Item: {self.m_item}")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def getSparseGraph(self, a=0, norm_type='row'):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + f'/s_pre_adj_mat_{norm_type}_{a}.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                save_flag = True
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                adj_mat = adj_mat + a * sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                if norm_type == 'row':
                    norm_adj = d_mat.dot(adj_mat)
                elif norm_type == 'col':
                    norm_adj = adj_mat.dot(d_mat)
                elif norm_type == 'sym':
                    d_inv = np.power(rowsum, -0.5).flatten()
                    d_inv[np.isinf(d_inv)] = 0.
                    d_mat = sp.diags(d_inv)
                    norm_adj = d_mat.dot(adj_mat)
                    norm_adj = norm_adj.dot(d_mat)
                # in this case, norm type is a float(e.g. 0.2)
                elif norm_type == 'un_norm':
                    norm_adj = adj_mat
                else:
                    p = float(norm_type)
                    q = 1 - p
                    d_inv_1 = np.power(rowsum, -p).flatten()
                    d_inv_2 = np.power(rowsum, -q).flatten()
                    d_mat_1 = sp.diags(d_inv_1)
                    d_mat_2 = sp.diags(d_inv_2)
                    norm_adj = d_mat_1.dot(adj_mat)
                    norm_adj = norm_adj.dot(d_mat_2)
                    
                    save_flag = True

                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"cost {end - s} s")
                if save_flag:
                    sp.save_npz(self.path + f'/s_pre_adj_mat_{norm_type}_{a}.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce()
            print("done adjacency matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class LP_Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, dataset, config):
        raw_dataset = Loader(dataset)
        self.n_user = raw_dataset.n_user
        self.m_item = raw_dataset.m_item
        self.teacher_k = config['teacher_k']
        sp_graph = raw_dataset.getSparseGraph(config['a'], config['norm_type'])

        indices = sp_graph.indices()
        values = sp_graph.values()

        size_dataset = raw_dataset.n_user + raw_dataset.m_item
        indices_2, values_2 = spspmm(indices, values, indices, values, size_dataset, size_dataset, size_dataset)
        indices_2, values_2 = self.drop_sparse_mat(indices_2, values_2, config['drop_ratio'])
        p2 = torch.sparse.FloatTensor(indices_2, values_2, (size_dataset, size_dataset)).coalesce()
        indices_2 = p2.indices()
        values_2 = p2.values()

        indices_3, values_3 = spspmm(indices, values, indices_2, values_2, size_dataset, size_dataset, size_dataset)
        # indices_3, values_3 = self.drop_sparse_mat(indices_3, values_3, 0.1)
        p3 = torch.sparse.FloatTensor(indices_3, values_3, (size_dataset, size_dataset)).coalesce().cpu()
        
        self.u_i_mat = self.get_u_i_mat(p3)
        # self.u_i_mat = self.get_u_i_mat(sp_graph)

        print(f"Total nnz entry is {len(self.u_i_mat.values())}")
        pre_matrix_result = Test_Sparse_Mat(raw_dataset, p3, config['k'])
        # pre_matrix_result = Test_Sparse_Mat(raw_dataset, sp_graph, config['k'])
        print(pre_matrix_result)
        # train or test

        rec_list, value_list = self.get_teacher_data(raw_dataset, indices_3, values_3, self.teacher_k)
        self.sample_prob = torch.tensor(value_list)
        
        self.dataset_name = raw_dataset.dataset_name
        self.split = False

        self.trainUniqueUsers = raw_dataset.trainUniqueUsers
        self.trainUser =  raw_dataset.trainUser
        self.trainItem = raw_dataset.trainItem
        self.traindataSize = raw_dataset.trainDataSize
        
        self.testDataSize = raw_dataset.testDataSize
        self.testUniqueUsers = raw_dataset.testUniqueUsers
        self.testUser = raw_dataset.testUser
        self.testItem = raw_dataset.testItem

        self.teacherUniqueUsers = self.trainUniqueUsers
        self.teacherUser = []
        self.teacherItem = []
        for u in range(self.n_user):
            self.teacherUser.extend([u] * len(rec_list))
            self.teacherItem.extend(rec_list[u].tolist())
        self.teacherUser = np.array(self.teacherUser)
        self.teacherItem = np.array(self.teacherItem)
        self.items_for_user_teacher = rec_list
        self._allPos_teacher = self.getUserPosItems_teacher(list(range(self.n_user)))

        self.UserItemNet = raw_dataset.UserItemNet
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self.user_D_normed = torch.FloatTensor(self.users_D / self.users_D.max())
        self.items_D_normed = torch.FloatTensor(self.items_D / self.items_D.max())
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
    
    def get_teacher_data(self, dataset, indices, values, K):
        '''
        Args:
            dataset: (Loader) raw_dataset
            indices: (torch.LongTensor) 
            values: (torch.FloatTensor)
            K: (int)
        '''
        valid_idx = (indices[0] < dataset.n_user) & (indices[1] >= dataset.n_user)
        valid_indices = indices[:, valid_idx]
        valid_indices[1] = valid_indices[1] - dataset.n_user

        valid_values = values[valid_idx]

        row_csr = valid_indices.numpy()[0]
        col_csr = valid_indices.numpy()[1]
        val_csr = valid_values.numpy()

        csr_mat = csr_matrix((val_csr, (row_csr, col_csr)), shape=(dataset.n_user, dataset.m_item))

        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(dataset.allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)

        csr_mat[exclude_index, exclude_items] = 0
        csr_mat.eliminate_zeros()

        rec_list = []
        value_list = []
        for user in range(dataset.n_user):
            row = csr_mat.getrow(user)
            key = row.indices
            val = row.data

            length = len(val)
            iter_obj = [(val[x], key[x]) for x in range(len(val))]
  
            if length < K:
                if length > 0:
                    rec_list_u = sorted(iter_obj, key=lambda x: x[0], reverse=True)
                    item_list_u = [x[1] for x in rec_list_u]
                    value_list_u = [x[0] for x in rec_list_u]
                else:
                    item_list_u = [random.randint(0, dataset.m_item - 1)]
                    value_list_u = [1]
                    length += 1
                rand_num = K - length
                for _ in range(rand_num):
                    item_list_u.append(random.randint(0, dataset.m_item - 1))
                    value_list_u.append(min(value_list_u))
            else:
                rec_list_u = nlargest(K, iter_obj, key=lambda x: x[0])
                item_list_u = [x[1] for x in rec_list_u]
                value_list_u = [x[0] for x in rec_list_u]

            rec_list.append(item_list_u)
            value_list.append(value_list_u)
        rec_list = np.array(rec_list)
        value_list = np.array(value_list)

        return rec_list, value_list


    def drop_sparse_mat(self, indices, values, ratio: float):
        '''
        Args:
            indices: (torch.LongTensor) 
            values: (torch.FloatTensor)
            ratio: (float)
        Returns:
            indices: (torch.LongTensor)
            value: (torch.FloatTensor)
        '''
        k = int(len(values) * ratio)
        _, idx = torch.topk(values, k)
        # idx, _ = idx.sort()
        indices = indices[:, idx]
        values = values[idx]

        return indices, values
    
    def get_u_i_mat(self, mat):
        '''
        Args:
            mat: (torch.sparse.FloatTensor) 
        Returns:
            u_i_mat: (torch.sparse.FloatTensor) 
        '''
        valid_idx = (mat.indices()[0] < self.n_user) & (mat.indices()[1] >= self.n_user)
        valid_indices = mat.indices()[:, valid_idx]
        valid_values = mat.values()[valid_idx]

        valid_indices[1] = valid_indices[1] - self.n_user

        u_i_mat = torch.sparse.FloatTensor(valid_indices,valid_values, (self.n_user, self.m_item)).coalesce()

        return u_i_mat

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].indices)
        return posItems
    
    def getUserPosItems_teacher(self, users):
        return self.items_for_user_teacher[users]