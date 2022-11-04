from heapq import nlargest
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def UniformSample_original_python(dataset, neg_k):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = dataset.trainDataSize
    users = dataset.trainUser
    pos_items = dataset.trainItem
    # weights = dataset.trainWeight
    neg_items = np.random.choice(dataset.m_items, size=(user_num, neg_k))
    # S = np.concatenate([users.reshape(-1, 1), weights.reshape(-1, 1), pos_items.reshape(-1, 1), neg_items], axis=1)
    S = np.concatenate([users.reshape(-1, 1), pos_items.reshape(-1, 1), neg_items], axis=1)
    return S

def UniformSample_teacher(dataset, neg_k):
    pos_items = []
    k = int(dataset.traindataSize / dataset.n_user) + 1
    sample_idx = torch.multinomial(dataset.sample_prob, k, replacement=True)
    pos_items = torch.LongTensor(dataset.items_for_user_teacher)[torch.LongTensor(list(range(dataset.n_user))), sample_idx.T].T.view(-1)
    users = np.arange(dataset.n_users).reshape(-1, 1).repeat(k, axis=1).reshape(1, -1)[0]
    users = users[:dataset.traindataSize]
    pos_items = pos_items[:dataset.traindataSize].numpy()
    user_num = len(users)
    neg_items = np.random.choice(dataset.m_items, size=(user_num, neg_k))
    S = np.concatenate([users.reshape(-1, 1), pos_items.reshape(-1, 1), neg_items], axis=1)
    return S
#--------------------------------------------------------------------------------------------------------#

def Test_Sparse_Mat(dataset, ratings, k):
    '''
    Args:
        dataset: (Loader object) defined in dataloader.py
        ratings: (torch.sparse.FloatTensor) object
        k: (int)
    Returns:
        A dict of recall@k, precision@k and ndcg@k
        example:
        {
            'recall': 0.12,
            'precision': 0.11,
            'ndcg': 0.13
        }
    '''

    result = {}

    rec_list = []
    dense_graph = ratings.cpu().to_dense()[:dataset.n_user, dataset.n_user:]
    exclude_index = []
    exclude_items = []
    for range_i, items in enumerate(dataset.allPos):
        exclude_index.extend([range_i] * len(items))
        exclude_items.extend(items)
    dense_graph[exclude_index, exclude_items] = -(1<<10)
    _, rec_list = torch.topk(dense_graph, k)

    result['recall'] = recallATk(rec_list, dataset, k)
    result['precision'] = precisionATk(rec_list, dataset, k)
    result['ndcg'] = ndcgATk(rec_list, dataset, k)

    return result

def recallATk(rec_list, dataset, k):
    '''
    Args: 
        rec_list: (Tensor) of recommended items for each users
        dataset: (Loader object) defined in dataloader.py
        k: (int)
    Returns:
        recall@k
    '''
    rec_list = rec_list.numpy().tolist()
    recall = 0.
    for u in range(dataset.n_user):
        rec_for_u = set(rec_list[u])
        truth_for_u = set(dataset.testDict[u])
        recall += len(rec_for_u & truth_for_u) / len(truth_for_u)

    return recall / dataset.n_user

def precisionATk(rec_list, dataset, k):
    '''
    Args: 
        rec_list: (Tensor) of recommended items for each users
        dataset: (Loader object) defined in dataloader.py
        k: (int)
    Returns:
        precision@k
    '''

    rec_list = rec_list.numpy().tolist()
    precision = 0.
    for u in range(dataset.n_user):
        rec_for_u = set(rec_list[u])
        truth_for_u = set(dataset.testDict[u])
        precision += len(rec_for_u & truth_for_u) / k

    return precision / dataset.n_user

def ndcgATk(rec_list, dataset, k):
    '''
    Args: 
        rec_list: (Tensor) of recommended items for each users
        dataset: (Loader object) defined in dataloader.py
        k: (int)
    Returns:
        ndcg@k
    '''
    ndcg = 0.
    rec_list = rec_list.numpy()
    for u in range(dataset.n_user):
        rec_for_u = rec_list[u]
        dcg_list = np.zeros(k)

        length = k if k <= len(dataset.testDict[u]) else len(dataset.testDict[u])
        dcg_list = np.zeros(k)
        idcg_list = np.zeros(k)
        idcg_list[:length] = 1
        for i in range(len(rec_for_u)):
            if rec_for_u[i] in dataset.testDict[u]:
                dcg_list[i] = 1
        idcg = np.sum(idcg_list * 1./np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg_list * 1./np.log2(np.arange(2, k + 2)))
        ndcg_u = 0 if dcg == 0 else dcg / idcg
        ndcg += ndcg_u
    return ndcg / dataset.n_user


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================