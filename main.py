import time
import numpy as np
import tensorflow as tf

from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
from prepare_data import random_select
from scipy import sparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


dataset = 'drugbank1'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
checkpt_file1 = 'test_args/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)#test model args
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 50
patience = 25
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi
#model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_dblp(path='data\\2SIDES_v4\MAT\drugbank0.mat'):
#def load_data_dblp(path='data\\case_study\\MAT\drugbank0.mat'):
#def load_data_dblp(path='test_drugbank.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    print(N)
    # print("-----------------------asdf-----------------------------------")
    # print(data['PTP'] - np.eye(N))
    # print("------------------------------------------------------------")
    #rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
    #rownetworks = [data['IDI'] - np.eye(N), data['IEI'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
    row = data["row"][0]
    col = data["col"][0]
    data1 = data["data"][0]
    # print("-----------------------asdf-----------------------------------")
    # print(data['PTP'] - np.eye(N))
    # print("------------------------------------------------------------")
    # rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
    # rownetworks = [data['IDI'] - np.eye(N), data['IEI'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
    IDI = sparse.coo_matrix((data1, (row, col)), shape=(4480, 4480), dtype='uint8')
    #case study
    #IDI = sparse.coo_matrix((data1, (row, col)), shape=(2482, 2482), dtype='uint8')
    rownetworks = [IDI - np.eye(N, dtype='uint8'),
                   np.eye(N, dtype='uint8') - np.eye(N, dtype='uint8')]  # , data['PTP'] - np.eye(N)]
    #print()
    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask
    #return rownetworks, truefeatures, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()
if featype == 'adj':
    fea_list = adj_list



import scipy.sparse as sp



# nb_nodes = fea_list[0].shape[0]
# ft_size = fea_list[0].shape[1]
# nb_classes = y_train.shape[1]

nb_nodes = 4480
#case study
#nb_nodes = 2482

ft_size = 881
nb_classes = 2


# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]
cnt = 1
print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        # ftr_in_list = tf.placeholder(dtype=tf.float32,
        #                               shape=(batch_size, nb_nodes, ft_size),
        #                               name='ftr_in_1')
        #
        # bias_in_list = tf.placeholder(dtype=tf.float32,
        #                                shape=(batch_size, nb_nodes, nb_nodes),
        #                                name='bias_in_1')

        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward

    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)
    # logits, final_embedding= model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
    #                                                    attn_drop, ffd_drop,
    #                                                    bias_mat=bias_in_list,
    #                                                    hid_units=hid_units, n_heads=n_heads,
    #                                                    residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    tlss_mn = np.inf
    tacc_mx = 0.0
    curr_step = 0

    #step_num_tr = 0
    #step_num_val = 0

    path = "data\\2SIDES_v4\MAT"
    fileList = os.listdir(path)
    fileList.sort(key=lambda x: int(x[8:-4]))  # 按编号取文件

    with tf.Session(config=config) as sess:
        # sess.run(init_op)
        #
        # train_loss_avg = 0
        # train_acc_avg = 0
        # val_loss_avg = 0
        # val_acc_avg = 0
        #
        #
        # for filename in fileList:
        #     train_loss_lst = []
        #     train_acc_lst = []
        #     val_loss_lst = []
        #     val_acc_lst = []
        #     # path_train = "data\\result\\train"
        #     # path_val = "data\\result\\val"
        #     # path_test = "data\\result\\test"
        #     path_train = "data\\gat_result\\train"
        #     path_val = "data\\gat_result\\val"
        #     path_test = "data\\gat_result\\test"
        #     if cnt >= 1:
        #         name = os.path.join(path, filename)  # file path
        #         adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp(name)
        #         fea_list = [fea[np.newaxis] for fea in fea_list]
        #         adj_list = [adj[np.newaxis] for adj in adj_list]
        #         y_train = y_train[np.newaxis]
        #         y_val = y_val[np.newaxis]
        #         y_test = y_test[np.newaxis]
        #         train_mask = train_mask[np.newaxis]
        #         val_mask = val_mask[np.newaxis]
        #         test_mask = test_mask[np.newaxis]
        #
        #         biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]
        #     for epoch in range(nb_epochs):
        #         tr_step = 0
        #
        #         tr_size = fea_list[0].shape[0]
        #         # ================   training    ============
        #         name_acc_t = "acc_data" + str(cnt)
        #         train_name_acc = os.path.join(path_train,"acc",name_acc_t)
        #         name_loss_t = "loss_data" + str(cnt)
        #         train_name_loss = os.path.join(path_train, "loss", name_loss_t)
        #         # print(ftr_in_list)
        #         # print("==================")
        #         # print(fea_list)
        #         print(type(ftr_in_list))
        #         print(type(fea_list))
        #         while tr_step * batch_size < tr_size:
        #             fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
        #                    for i, d in zip(ftr_in_list, fea_list)}
        #             fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
        #                    for i, d in zip(bias_in_list, biases_list)}
        #             # fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
        #             #        for i, d in zip(float(ftr_in_list.eval(session = sess)), fea_list)}
        #             # fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
        #             #        for i, d in zip(float(bias_in_list.eval(session = sess)), biases_list)}
        #             fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
        #                    msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
        #                    is_train: True,
        #                    attn_drop: 0.6,
        #                    ffd_drop: 0.6}
        #             fd = fd1
        #             fd.update(fd2)
        #             fd.update(fd3)
        #             _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy],#, att_val],
        #                                                                feed_dict=fd)
        #             train_loss_avg += loss_value_tr
        #             train_acc_avg += acc_tr
        #             tr_step += 1
        #             #step_num_tr += 1
        #         train_loss_lst.append(train_loss_avg / tr_step)
        #         train_acc_lst.append(train_acc_avg / tr_step)
        #
        #         vl_step = 0
        #         vl_size = fea_list[0].shape[0]
        #         # =============   val       =================
        #         name_acc_v = "acc_data" + str(cnt)
        #         val_name_acc = os.path.join(path_val, "acc", name_acc_v)
        #         name_loss_v = "loss_data" + str(cnt)
        #         val_name_loss = os.path.join(path_val, "loss", name_loss_v)
        #         while vl_step * batch_size < vl_size:
        #             # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
        #             fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
        #                    for i, d in zip(ftr_in_list, fea_list)}
        #             fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
        #                    for i, d in zip(bias_in_list, biases_list)}
        #             fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
        #                    msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
        #                    is_train: False,
        #                    attn_drop: 0.0,
        #                    ffd_drop: 0.0}
        #
        #             fd = fd1
        #             fd.update(fd2)
        #             fd.update(fd3)
        #             loss_value_vl, acc_vl = sess.run([loss, accuracy],
        #                                              feed_dict=fd)
        #             val_loss_avg += loss_value_vl
        #             val_acc_avg += acc_vl
        #             vl_step += 1
        #             #step_num_val += 1
        #         # import pdb; pdb.set_trace()
        #         print('Number: {},Epoch: {}, att_val: {}'.format(cnt,epoch, np.mean(att_val_train, axis=0)))
        #
        #         print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
        #               (train_loss_avg / tr_step, train_acc_avg / tr_step,
        #                val_loss_avg / vl_step, val_acc_avg / vl_step))
        #         val_loss_lst.append(val_loss_avg / vl_step)
        #         val_acc_lst.append(val_acc_avg / vl_step)
        #         # if val_acc_avg / vl_step > 1:
        #         #     cnt = cnt + 1
        #         #     continue
        #         if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
        #         #if val_acc_avg / vl_step > vacc_mx: #or val_loss_avg / step_num_val <= vlss_mn:
        #             if val_acc_avg / vl_step <= 1:
        #
        #                 if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
        #                     vacc_early_model = val_acc_avg / vl_step
        #                     vlss_early_model = val_loss_avg / vl_step
        #                     saver.save(sess, checkpt_file)
        #                     vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
        #                     vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
        #                     curr_step = 0
        #         else:
        #             curr_step += 1
        #             if curr_step == patience:
        #                 print('Early stop! Min loss: ', vlss_mn,
        #                       ', Max accuracy: ', vacc_mx)
        #                 print('Early stop model validation loss: ',
        #                       vlss_early_model, ', accuracy: ', vacc_early_model)
        #                 cnt = cnt + 1
        #                 train_loss_avg = 0
        #                 train_acc_avg = 0
        #                 val_loss_avg = 0
        #                 val_acc_avg = 0
        #                 curr_step = 0
        #                 break
        #         if epoch == nb_epochs - 1:
        #             cnt = cnt + 1
        #             curr_step = 0
        #         train_loss_avg = 0
        #         train_acc_avg = 0
        #         val_loss_avg = 0
        #         val_acc_avg = 0
        #
        #     train_loss_lst = np.array(train_loss_lst)
        #     train_acc_lst = np.array(train_acc_lst)
        #     val_loss_lst = np.array(val_loss_lst)
        #     val_acc_lst = np.array(val_acc_lst)
        #
        #     np.save(train_name_acc,train_acc_lst)
        #     np.save(train_name_loss, train_loss_lst)
        #     np.save(val_name_acc, val_acc_lst)
        #     np.save(val_name_loss, val_loss_lst)
        #     """# 保存
        #     import numpy as np
        #     a=np.array(a)
        #     np.save('a.npy',a)   # 保存为.npy格式
        #     # 读取
        #     a=np.load('a.npy')
        #     a=a.tolist()"""


        ############test############################
        path_test = "data\\result\\test"
        saver.restore(sess, checkpt_file1)
        print('load model from : {}'.format(checkpt_file1))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0
        test_loss_lst = []
        test_acc_lst = []
        test_name_acc = os.path.join(path_test, "acc")
        test_name_loss = os.path.join(path_test, "loss")
        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
        # while ts_step * batch_size < 1:
        #     # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
        #     fd1 = {i: d[-1:]
        #            for i, d in zip(ftr_in_list, fea_list)}
        #     fd2 = {i: d[-1:]
        #            for i, d in zip(bias_in_list, biases_list)}
        #     fd3 = {lbl_in: y_test[-1:],
        #            msk_in: test_mask[-1:],
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)
        if ts_acc / ts_step > tacc_mx:
            saver.save(sess, checkpt_file1)
        tacc_mx = np.max((ts_acc / ts_step, tacc_mx))
        tlss_mn = np.min((ts_loss / ts_step, tlss_mn))
        test_loss_lst.append(ts_loss / ts_step)
        test_acc_lst.append(ts_acc / ts_step)
        np.save(test_name_loss, test_loss_lst)
        np.save(test_name_acc, test_acc_lst)


        print('start knn, kmean.....')
        xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]
        """
        for sample in xx:
            print(sample)
        """
  
        from numpy import linalg as LA

        # xx = xx / LA.norm(xx, axis=1)
        yy = y_test[test_mask[-1:]]

        print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        from jhyexp import my_KNN, my_Kmeans#, my_TSNE, my_Linear

        #my_KNN(xx, yy)
        for i in range(2,11):
            print("k=",i)
            my_KNN(xx, yy,k=i)
        print("10 fold")
        for i in range(2,11):
            print("k=",i)
            my_KNN(xx,yy,k=i,split_list=[0.1, 0.9],time=10)
        #case study

        #my_Kmeans(xx, yy,k=1)
        sess.close()
