import time
import torch
from utils import timer
import utils
import numpy as np
import multiprocessing

def Softmax_train_teacher(dataset, recommend_model, epoch, config, opt, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()

    start_time = time.time()
    with timer(name="Sample"):
        S = utils.UniformSample_original_python(dataset, neg_k)

    users = torch.Tensor(S[:, 0]).long()
    items = torch.Tensor(S[:, 1:]).long()

    S_teacher = utils.UniformSample_teacher(dataset, neg_k)
    users_teacher = torch.Tensor(S_teacher[:, 0]).long()
    items_teacher = torch.Tensor(S_teacher[:, 1:]).long()

    users, items, users_teacher, items_teacher = utils.shuffle(users, items, users_teacher, items_teacher)
    total_batch = len(users) // config['batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_items,
          batch_users_teacher,
          batch_items_teacher)) in enumerate(utils.minibatch(users,
                                                   items,
                                                   users_teacher, 
                                                   items_teacher,
                                                   batch_size=config['batch_size'])):
        batch_users = batch_users.to(config['device'])
        batch_items = batch_items.to(config['device'])
        batch_users_teacher = batch_users_teacher.to(config['device'])
        batch_items_teacher = batch_items_teacher.to(config['device'])
        sm_loss_ranking, reg_loss_ranking = recommend_model.softmax_loss(batch_users, batch_items)
        sm_loss_teacher, reg_loss_teacher = recommend_model.softmax_loss(batch_users_teacher, batch_items_teacher, weights=False)
        reg_loss = (reg_loss_ranking + reg_loss_teacher) * config['decay']
        loss = reg_loss + config['alpha'] * sm_loss_ranking + config['beta'] * sm_loss_teacher
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss += loss.cpu().item()
        if config['tensorboard']:
            w.add_scalar(f'Loss/loss', loss, epoch * int(len(users) / config['batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    end_time = time.time()
    return f"loss{aver_loss:.3f} - time:{(end_time - start_time):.3f}"

def Test(dataset, Recmodel, epoch, config, w=None, multicore=0):
    u_batch_size = config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config['topks'])
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(config['topks'])),
               'recall': np.zeros(len(config['topks'])),
               'ndcg': np.zeros(len(config['topks']))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config['device'])

            rating = Recmodel.getUsersRating(batch_users_gpu).cpu()
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X, config['topks'])
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, config['topks']))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if config['tensorboard']:
            topk = config['topks']
            w.add_scalars(f'Test/Recall@{topk}',
                          {str(config['topks'][i]): results['recall'][i] for i in range(len(config['topks']))}, epoch)
            w.add_scalars(f'Test/Precision@{topk}',
                          {str(config['topks'][i]): results['precision'][i] for i in range(len(config['topks']))}, epoch)
            w.add_scalars(f'Test/NDCG@{topk}',
                          {str(config['topks'][i]): results['ndcg'][i] for i in range(len(config['topks']))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}