import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import scipy.io as scio
import random
import pickle
import time

def feature_tsne_fea(x, x_mean, attr_x):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    xs = torch.cat([x, x_mean, attr_x], dim=0)
    all_data = xs.cpu().detach().numpy()
    tsne = tsne.fit_transform(all_data)
    return tsne


def plot_feature_tsne_fea(x, y, x_mean, y_mean, attr_x, attr_y, fig_path):
    if len(np.unique(y_mean)) == 2:
        colors = ['b', 'r']
        for index in range(len(colors)):
            sel_index = np.where(y == index)
            sel_x = x[sel_index[0], :]
            if index == 0:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                            label='x(without attr)')
            else:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                            label='x(with attr)')
        # plt.scatter(x[:, 0], x[:, 1], 1, marker='.', c=y[:], label='x')
        for index in range(len(colors)):
            sel_index = np.where(y_mean == index)
            sel_x = x_mean[sel_index[0], :]
            if index == 0:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=100, alpha=0.2, marker='*',
                            label='class prototype(without attr)')
            else:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=100, alpha=0.2, marker='*',
                            label='class prototype(with attr)')
        # plt.scatter(x_mean[:, 0], x_mean[:, 1], 200, marker='*', c=y_mean[:], label='mean')
        plt.scatter(attr_x[:, 0], attr_x[:, 1], 100, marker='s', c=colors[1], label='attributes prototype')
        # plt.legend(loc='upper right')
        plt.axis('off')
        plt.savefig(fig_path, dpi=600)
        plt.show()
        plt.close()
    else:
        colors = ['b', 'r', 'g', 'm']
        for index in range(len(colors)):
            sel_index = np.where(y == index)
            sel_x = x[sel_index[0], :]
            if index == 0:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                            label='x(without attr)')
            else:
                if index == 1:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                                label='train x(with attr)')
                elif index == 2:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                                label='val x(with attr)')
                else:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=1, alpha=0.2, marker='.',
                                label='test x(with attr)')
        # plt.scatter(x[:, 0], x[:, 1], 1, marker='.', c=y[:], label='x')
        for index in range(len(colors)):
            sel_index = np.where(y_mean == index)
            sel_x = x_mean[sel_index[0], :]
            if index == 0:
                plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=10, alpha=0.8, marker='*',
                            label='class prototype(without attr)')
            else:
                if index == 1:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=10, alpha=0.8, marker='*',
                                label='train class prototype(with attr)')
                elif index == 2:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=10, alpha=0.8, marker='*',
                                label='val class prototype(with attr)')
                else:
                    plt.scatter(sel_x[:, 0], sel_x[:, 1], c=colors[index], cmap='brg', s=10, alpha=0.8, marker='*',
                                label='test class prototype(with attr)')

        # plt.scatter(x_mean[:, 0], x_mean[:, 1], 200, marker='*', c=y_mean[:], label='mean')
        plt.scatter(attr_x[:, 0], attr_x[:, 1], 20, marker='s', c=colors[1], alpha=0.8, label='attributes prototype')
        # plt.legend(loc='upper right')
        plt.axis('off')
        plt.savefig(fig_path, dpi=600)
        plt.show()
        plt.close()


def feature_tsne_with_attr(x, y, x_mean, y_mean, attr_x, attr_y, fig_path):
    x_size = x.shape[0]
    x_mean_size = x_mean.shape[0]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    xs = torch.cat([x, x_mean, attr_x], dim=0)
    ys = torch.cat([y, y_mean, attr_y], dim=0)
    all_data = xs.cpu().detach().numpy()
    tsne = tsne.fit_transform(all_data)
    plot_embedding_with_attr(tsne, ys.cpu().detach().numpy(), x_size, x_mean_size)
    plt.savefig(fig_path, dpi=600)
    plt.show()
    plt.close()


def plot_embedding_with_attr(xs, ys, x_size, x_mean_size):
    x = xs[:x_size, :]
    x_mean = xs[x_size:x_size + x_mean_size, :]
    y = ys[:x_size]
    y_mean = ys[x_size:x_size + x_mean_size]
    x_attr = xs[x_size + x_mean_size:, :]
    y_attr = ys[x_size + x_mean_size:]

    plt.scatter(x[:, 0], x[:, 1], 1, marker='.', c=y[:], label='x')
    plt.scatter(x_mean[:, 0], x_mean[:, 1], 200, marker='*', c=y_mean[:], label='mean')
    plt.scatter(x_attr[:, 0], x_attr[:, 1], 200, marker='s', c=y_attr[:], label='attributes')
    # plt.legend(loc='upper right')


def feature_tsne(x, y, x_mean, y_mean, fig_path):
    x_size = x.shape[0]
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=1000)
    xs = torch.cat([x, x_mean], dim=0)
    ys = torch.cat([y, y_mean], dim=0)
    all_data = xs.cpu().detach().numpy()
    tsne = tsne.fit_transform(all_data)
    plot_embedding(tsne, ys.cpu().detach().numpy(), x_size)
    plt.axis('off')
    plt.savefig(fig_path, dpi=600, format=fig_path[-3:])
    # plt.show()
    plt.close()
    print(fig_path)
    scio.savemat(fig_path[:-4] + '1.mat',
                 {'data': xs.cpu().detach().numpy(), 'tsne': tsne, 'label': ys.cpu().detach().numpy(),
                  'number': x_size})


def feature_tsne_dynamics(x, y, x_mean, y_mean, fig_path):
    x_size = x.shape[0]
    # tsne = TSNE(perplexity=80, n_components=2, init='pca', n_iter=500, metric='cosine')
    tsne = TSNE(perplexity=70, n_components=2, init='pca', n_iter=500, metric='euclidean')
    xs = torch.cat([x, x_mean], dim=0)
    ys = torch.cat([y, y_mean], dim=0)
    all_data = xs.cpu().detach().numpy()
    tsne = tsne.fit_transform(all_data)
    ## split data
    data_point = tsne[:x_size, :]
    data_point_y = y.cpu().detach().numpy()

    dynamics_start_point = tsne[x_size:x_size + 5, :]
    dynamics_start_point_y = y_mean.cpu().detach().numpy()[:5]

    dynamics_stop_point = tsne[-5:, :]
    dynamics_stop_point_y = y_mean.cpu().detach().numpy()[-5:]

    dynamics_mid_point = []
    dynamics_mid_point_y = []
    for i in range(5):
        path = []
        path_y = []
        for j in range(30):
            path.append(tsne[x_size + j * 5 + i, :])
            path_y.append(y_mean.cpu().detach().numpy()[j * 5 + i])
        path = np.stack(path, axis=0)
        path_y = np.stack(path_y, axis=0)
        dynamics_mid_point.append(path)
        dynamics_mid_point_y.append(path_y)
    dynamics_mid_point = np.stack(dynamics_mid_point, axis=0)
    dynamics_mid_point_y = np.stack(dynamics_mid_point_y, axis=0)

    plot_embedding_dynamic(data_point, data_point_y,
                           dynamics_start_point, dynamics_start_point_y,
                           dynamics_stop_point, dynamics_stop_point_y,
                           dynamics_mid_point, dynamics_mid_point_y)
    # plt.axis('off')
    plt.savefig(fig_path[:-3] + 'pdf', dpi=600, format='pdf')
    plt.savefig(fig_path[:-3] + 'jpg', dpi=600, format='jpg')
    # plt.show()
    plt.close()
    print(fig_path)
    scio.savemat(fig_path[:-4] + '1.mat',
                 {'data': xs.cpu().detach().numpy(), 'tsne': tsne, 'label': ys.cpu().detach().numpy(),
                  'number': x_size})


def feature_tsne_no_center(x, y, fig_path,upons=None):
    x_size = len(x)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    xs = x
    ys = y
    all_data = xs
    tsne = tsne.fit_transform(all_data)
    plot_embedding(tsne, ys, x_size)
    plt.axis('off')
    time_ = time.time()
    plt.savefig(os.path.join(fig_path,str(time_)+'.png'), dpi=600)
    print('saveto{0}'.format(os.path.join(fig_path,str(time_)+'.png')))
    # plt.show()
    plt.close()
    print(fig_path)
    scio.savemat(fig_path[:-4] + '1.mat',
                 {'data': xs, 'tsne': tsne, 'label': ys,
                  'number': x_size})

def feature_tsne_vq_vis(fea_x, fea_y,weight_x,weight_y,connected_x,connected_y, fig_path):
    fea_x_size = len(fea_x)
    weight_x_size = len(weight_x)
    connected_x_size = len(connected_x)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    all_data_fea = np.array(fea_x)
    all_data_w = np.array(weight_x)
    all_data_con = np.array(connected_x)
    tsne_fea = tsne.fit_transform(all_data_fea)
    tsne_w = tsne.fit_transform(all_data_w)
    tsne_con = tsne.fit_transform(all_data_con)
    plot_embedding_vis(tsne_fea, fea_y,tsne_w,weight_y,tsne_con,connected_y, fea_x_size,weight_x_size,connected_x_size)
    plt.axis('off')
    time_ = time.time()
    plt.savefig(os.path.join(fig_path,str(time_)+'.png'), dpi=600)
    print('saveto{0}'.format(os.path.join(fig_path,str(time_)+'.png')))
    # plt.show()
    plt.close()
    print(fig_path)

def plot_embedding(xs, ys, x_size):
    x = xs[:x_size, :]
    x_mean = xs[x_size:, :]
    y = ys[:x_size]
    y_mean = ys[x_size:]

    plt.scatter(x[:, 0], x[:, 1], 10, marker='.', c='black', label='x')
    plt.scatter(x_mean[:, 0], x_mean[:, 1], 10, marker='+', c=y_mean[:], label='mean')
    # plt.legend(loc='upper right')

def plot_embedding_vis(fea_x, fea_y,weight_x,weight_y,connected_x,connected_y,size1,size2,size3 ):
    # 画出 feature0 的第一个点
    plt.scatter(fea_x[0, 0], fea_x[0, 1], 10, marker='.', c='yellow', label='x')
    # 画出 feature剩余的点
    for i in range(1,size1):
        plt.scatter(fea_x[i, 0], fea_x[i, 1], 10, marker='.', c='orange', label='x')
    # codebook
    for i in range(size2):
        if weight_y[i] in connected_y:
            continue
        else:
            plt.scatter(weight_x[i, 0], weight_x[i, 1], 10, marker='+', c='blue', label='mean')
    #connected
    for i in range(0,size3-1):
            plt.scatter(connected_x[i, 0], connected_x[i, 1], 10, marker='+', c='Cyan', label='mean')
    #choose one
    plt.scatter(connected_x[size3-1, 0], connected_x[size3-1, 1], 10, marker='+', c='black', label='mean')
    # plt.legend(loc='upper right')

def plot_embedding_upon(xs, ys, x_size, upons):
    x = xs[:x_size, :]
    x_mean = xs[x_size:, :]
    y = ys[:x_size]
    y_mean = ys[x_size:]
    for i in range(0,x_size,9):
        if i not in upons:
            plt.scatter(x[i, 0], x[i, 1], 10, marker='o', c='#66CDAA', label='x',alpha=0.5)
    for upon in upons:
        plt.scatter(x[upon, 0], x[upon, 1], 30, marker='o', c=y[upon], label='mean',alpha=0.5,edgecolor=y[upon])


def plot_embedding_dynamic(data_point, data_point_y,
                           dynamics_start_point, dynamics_start_point_y,
                           dynamics_stop_point, dynamics_stop_point_y,
                           dynamics_mid_point, dynamics_mid_point_y):
    # from collections import OrderedDict
    # cmaps = OrderedDict()
    plt.figure(figsize=(5, 4))
    cmap = list()
    cmaps = [
        'g', 'b', 'k', 'r', 'y']
    p0 = plt.scatter([data_point[idx, 0] for idx, item in enumerate(data_point_y[:]) if item == 0],
                     [data_point[idx, 1] for idx, item in enumerate(data_point_y[:]) if item == 0],
                     130,
                     marker='.',
                     c=[cmaps[item] for idx, item in enumerate(data_point_y[:]) if item == 0],
                     label='0',
                     alpha=0.3)
    p1 = plt.scatter([data_point[idx, 0] for idx, item in enumerate(data_point_y[:]) if item == 1],
                     [data_point[idx, 1] for idx, item in enumerate(data_point_y[:]) if item == 1],
                     130,
                     marker='.',
                     c=[cmaps[item] for idx, item in enumerate(data_point_y[:]) if item == 1],
                     label='1',
                     alpha=0.3)
    p2 = plt.scatter([data_point[idx, 0] for idx, item in enumerate(data_point_y[:]) if item == 2],
                     [data_point[idx, 1] for idx, item in enumerate(data_point_y[:]) if item == 2],
                     130,
                     marker='.',
                     c=[cmaps[item] for idx, item in enumerate(data_point_y[:]) if item == 2],
                     label='2',
                     alpha=0.3)
    p3 = plt.scatter([data_point[idx, 0] for idx, item in enumerate(data_point_y[:]) if item == 3],
                     [data_point[idx, 1] for idx, item in enumerate(data_point_y[:]) if item == 3],
                     130,
                     marker='.',
                     c=[cmaps[item] for idx, item in enumerate(data_point_y[:]) if item == 3],
                     label='3',
                     alpha=0.3)
    p4 = plt.scatter([data_point[idx, 0] for idx, item in enumerate(data_point_y[:]) if item == 4],
                     [data_point[idx, 1] for idx, item in enumerate(data_point_y[:]) if item == 4],
                     130,
                     marker='.',
                     c=[cmaps[item] for item in data_point_y[:] if item == 4],
                     label='4',
                     alpha=0.3)
    p5 = plt.scatter([dynamics_start_point[idx, 0] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 0],
                     [dynamics_start_point[idx, 1] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 0],
                     140,
                     marker='s',
                     c=[cmaps[item] for item in dynamics_start_point_y[:] if item == 0],
                     label='mean')
    p6 = plt.scatter([dynamics_start_point[idx, 0] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 1],
                     [dynamics_start_point[idx, 1] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 1],
                     140,
                     marker='s',
                     c=[cmaps[item] for item in dynamics_start_point_y[:] if item == 1],
                     label='mean')
    p7 = plt.scatter([dynamics_start_point[idx, 0] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 2],
                     [dynamics_start_point[idx, 1] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 2],
                     140,
                     marker='s',
                     c=[cmaps[item] for item in dynamics_start_point_y[:] if item == 2],
                     label='mean')
    p8 = plt.scatter([dynamics_start_point[idx, 0] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 3],
                     [dynamics_start_point[idx, 1] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 3],
                     140,
                     marker='s',
                     c=[cmaps[item] for item in dynamics_start_point_y[:] if item == 3],
                     label='mean')
    p9 = plt.scatter([dynamics_start_point[idx, 0] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 4],
                     [dynamics_start_point[idx, 1] for idx, item in enumerate(dynamics_start_point_y[:]) if item == 4],
                     140,
                     marker='s',
                     c=[cmaps[item] for item in dynamics_start_point_y[:] if item == 4],
                     label='mean')
    p10 = plt.scatter([dynamics_stop_point[idx, 0] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 0],
                      [dynamics_stop_point[idx, 1] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 0],
                      240,
                      marker='*',
                      c=[cmaps[item] for item in dynamics_stop_point_y[:] if item == 0],
                      label='mean')
    p11 = plt.scatter([dynamics_stop_point[idx, 0] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 1],
                      [dynamics_stop_point[idx, 1] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 1],
                      240,
                      marker='*',
                      c=[cmaps[item] for item in dynamics_stop_point_y[:] if item == 1],
                      label='mean')
    p12 = plt.scatter([dynamics_stop_point[idx, 0] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 2],
                      [dynamics_stop_point[idx, 1] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 2],
                      240,
                      marker='*',
                      c=[cmaps[item] for item in dynamics_stop_point_y[:] if item == 2],
                      label='mean')
    p13 = plt.scatter([dynamics_stop_point[idx, 0] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 3],
                      [dynamics_stop_point[idx, 1] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 3],
                      240,
                      marker='*',
                      c=[cmaps[item] for item in dynamics_stop_point_y[:] if item == 3],
                      label='mean')
    p14 = plt.scatter([dynamics_stop_point[idx, 0] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 4],
                      [dynamics_stop_point[idx, 1] for idx, item in enumerate(dynamics_stop_point_y[:]) if item == 4],
                      240,
                      marker='*',
                      c=[cmaps[item] for item in dynamics_stop_point_y[:] if item == 4],
                      label='mean')

    p15 = plt.plot(dynamics_mid_point[0, :, 0], dynamics_mid_point[0, :, 1], c=cmaps[0], linestyle='-', label='x1',
                   linewidth=1, alpha=0.5)
    p16 = plt.plot(dynamics_mid_point[1, :, 0], dynamics_mid_point[1, :, 1], c=cmaps[1], linestyle='-', label='x2',
                   linewidth=1, alpha=0.5)
    p17 = plt.plot(dynamics_mid_point[2, :, 0], dynamics_mid_point[2, :, 1], c=cmaps[2], linestyle='-', label='x3',
                   linewidth=1, alpha=0.5)
    p18 = plt.plot(dynamics_mid_point[3, :, 0], dynamics_mid_point[3, :, 1], c=cmaps[3], linestyle='-', label='x4',
                   linewidth=1, alpha=0.5)
    p19 = plt.plot(dynamics_mid_point[4, :, 0], dynamics_mid_point[4, :, 1], c=cmaps[4], linestyle='-', label='x5',
                   linewidth=1, alpha=0.5)

    # plot arrow
    # plt.quiver(dynamics_mid_point[0,15,0], dynamics_mid_point[0,15,1],
    #            dynamics_mid_point[0,16,1]-dynamics_mid_point[0,15,0], dynamics_mid_point[0,16,1]-dynamics_mid_point[0,15,1], angles='xy', scale_units='xy', scale=1)

    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    # legend = plt.legend(prop=f/ont)
    l = plt.legend([(p0, p1, p2, p3, p4), (p5, p6, p7, p8, p9), (p10, p11, p12, p13, p14)],
                   ['Query Samples', 'Initial Prototypes', 'Optimal Prototypes'],
                   scatterpoints=1,
                   numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best', handlelength=4, prop=font)
    import matplotlib as mpl
    # fm = mpl.font_manager
    # fm.get_cachedir()
    # mpl.rcParams['font.family'] = ['serif']
    # mpl.rcParams['font.serif'] = ['Times New Roman']
    # plt.rc('font', family='Times New Roman')
    plt.xticks(fontproperties='Times New Roman', size=17)
    plt.yticks(fontproperties='Times New Roman', size=17)
    plt.ylabel(r'$x_1$', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.xlabel(r'$x_2$', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.tight_layout()

    # plt.legend(loc='upper right')


def plot_all_data(target_tsne, y_spt, y_qry, pic_path, eposide_number, task_n, name):
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    path = pic_path + '/' + str(eposide_number) + '_' + str(task_n) + '_' + name + '.jpg'
    plt.title("Target space")
    plot_embedding(target_tsne, y_spt, y_qry)
    plt.savefig(path, dpi=600)
    # plt.show()
    plt.close()


def feature_tsne_task(embs, support, query, prototypes_init, boost_prototypes, mean_1, mean_2, prototypes,
                      prototypes_ture, average_prototypes, support_labels, query_labels, prototype_labels, fig_path):
    n_embs = embs.shape[0]
    n_support = support.shape[1]
    n_query = query.shape[1]
    n_prototypes_init = prototypes_init.shape[1]
    n_boost_prototypes = boost_prototypes.shape[1]
    n_mean_1 = mean_1.shape[1]
    n_mean_2 = mean_2.shape[1]
    n_prototypes = prototypes.shape[1]
    n_prototypes_ture = prototypes_ture.shape[1]
    n_average_prototypes = average_prototypes.shape[1]
    support = support.squeeze(dim=0)
    query = query.squeeze(dim=0)
    prototypes_init = prototypes_init.squeeze(dim=0)
    boost_prototypes = boost_prototypes.squeeze(dim=0)
    mean_1 = mean_1.squeeze(dim=0)
    mean_2 = mean_2.squeeze(dim=0)
    prototypes = prototypes.squeeze(dim=0)
    prototypes_ture = prototypes_ture.squeeze(dim=0)
    average_prototypes = average_prototypes.squeeze(dim=0)
    support_labels = support_labels.squeeze(dim=0)
    query_labels = query_labels.squeeze(dim=0)
    index = list(range(12000))
    random.shuffle(index)

    x = torch.cat([embs[-12000:, :][index[:1000], :],
                   support,
                   query,
                   prototypes_init,
                   boost_prototypes,
                   mean_1,
                   mean_2,
                   prototypes,
                   prototypes_ture,
                   average_prototypes], dim=0)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)

    all_data = x.cpu().detach().numpy()
    tsne = tsne.fit_transform(all_data)

    n_embs = 1000
    x_support = tsne[n_embs:n_embs + n_support, :]
    y_support = support_labels.cpu().detach().numpy()
    # plt.scatter(x_support[:, 0], x_support[:,1], 10, cmap='Set1', marker='.', c=y_support[:], label='$x$', alpha=0.5)

    x_query = tsne[n_embs + n_support:n_embs + n_support + n_query, :]
    x_query = np.concatenate([x_support, x_query], axis=0)
    y_query = query_labels.cpu().detach().numpy()
    y_query = np.concatenate([y_support, y_query], axis=0)
    plt.scatter(x_query[:, 0], x_query[:, 1], 2, cmap='Set1', marker='.', c=y_query[:], label='$x$', alpha=0.4)

    x_prototypes_init = tsne[n_embs + n_support + n_query:n_embs + n_support + n_query + n_prototypes_init, :]
    y_prototypes_init = prototype_labels.cpu().detach().numpy()
    plt.scatter(x_prototypes_init[:, 0], x_prototypes_init[:, 1], 80, cmap='Set1', marker='x', c=y_prototypes_init[:],
                label='$p_k$')

    x_boost_prototypes = tsne[
                         n_embs + n_support + n_query + n_prototypes_init:n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes,
                         :]
    y_boost_prototypes = prototype_labels.cpu().detach().numpy()
    plt.scatter(x_boost_prototypes[:, 0], x_boost_prototypes[:, 1], 80, cmap='Set1', marker='+',
                c=y_boost_prototypes[:], label='$\hat{p}_k$')

    # x_mean_1 = tsne[n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes:
    #                 n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1,:]
    # y_mean_1 = prototype_labels.cpu().detach().numpy()
    # plt.scatter(x_mean_1[:, 0], x_mean_1[:, 1], 50, marker='^', c=y_mean_1[:],
    #             label='mean_1')
    #
    # x_mean_2 = tsne[n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1:
    #                 n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2,:]
    # y_mean_2 = prototype_labels.cpu().detach().numpy()
    # plt.scatter(x_mean_2[:, 0], x_mean_2[:, 1], 50, marker='x', c=y_mean_2[:],
    #             label='mean_2')

    x_prototypes = tsne[n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2:
                        n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2 + n_prototypes,
                   :]
    y_prototypes = prototype_labels.cpu().detach().numpy()
    plt.scatter(x_prototypes[:, 0], x_prototypes[:, 1], 80, cmap='Set1', marker='s', c=y_prototypes[:],
                label='$\hat{p}\'_k$')

    x_prototypes_ture = tsne[
                        n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2 + n_prototypes:
                        n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2 + n_prototypes + n_prototypes_ture,
                        :]
    y_prototypes = prototype_labels.cpu().detach().numpy()
    plt.scatter(x_prototypes_ture[:, 0], x_prototypes_ture[:, 1], 120, cmap='Set1', marker='*', c=y_prototypes[:],
                label='$p^{real}_k$')

    # x_prototypes_ave = tsne[n_embs + n_support + n_query + n_prototypes_init + n_boost_prototypes + n_mean_1 + n_mean_2+n_prototypes+n_prototypes_ture:,:]
    # y_prototypes = prototype_labels.cpu().detach().numpy()
    # plt.scatter(x_prototypes_ave[:, 0], x_prototypes_ave[:, 1], 80, cmap='Set1', marker='^', c=y_prototypes[:],
    #             label='$\hat{p}\'_k$(MF)')
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 14})
    plt.axis('off')
    plt.savefig(fig_path, dpi=600)
    plt.show()
    plt.close()
    print(fig_path)


if __name__ == '__main__':
    with open('big_nlp_word_knowledge_clip.pkl', 'rb') as handle:
        graph = pickle.load(handle)
    # for item in graph:
    #     print(item)
    #     adj_code_book
    #     noun_code_book
    #     noun_code_book_vec
    #     adj_code_book_vec
    #     adj_code_book_num
    #     noun_code_book_num
    #     adj_noun_edges
    x = graph['adj_code_book']
    x = np.array(x)
    # print(len(x))
    x_ = []
    # for item in x:
    #     print(item)
    upon_index = []
    for index, item in enumerate(x):
        if item == 'yellow':
            x_.append('yellow')
            upon_index.append(index)
        elif item == "orange":
            x_.append('orange')
            upon_index.append(index)
        elif item == 'blue':
            x_.append("blue")
            upon_index.append(index)
            print(1)
        else:
            x_.append('Cyan')
    key = [i for i in range(1949)]
    key = np.array(key)
    y = graph['adj_code_book_vec']
    y = y.values()
    y = np.array(list(y))
    feature_tsne_no_center(x=y, y=x_, fig_path='adj.png',upons=upon_index)
