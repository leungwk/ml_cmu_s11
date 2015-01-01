import pandas as pd
import numpy as np
from nbayes import _cache
import matplotlib.pyplot as plt

data_dir = 'data/hw3/'
cache_dir = 'cache/'

def p_y_given_x(x, W):
    """
.. math::
    P(Y=k|X=x;W) = \frac{\exp(w_k^Tx)}{1 +\sum_{l=1}^{K-1} \exp(w_l^Tx)}\quad\text{for}\ k=1,...,K"""
    numer = np.exp(np.dot(W,x))
    denom = 1 +np.sum(np.exp(np.dot(W,x)))
    return 1.*numer/denom # this vector should sum to 1


def p_Y_given_X(X, W):
    """the matrix version of p_y_given_x()"""
    acc = []
    for r_idx, xval in X.iterrows():
        acc.append(p_y_given_x(xval, W))
    return np.array(acc)


def log_likelihood(X, y, W, lda):
    """
.. math::
    L(w_1,...,w_{K-1})
    &= \sum_{i=1}^n \ln P(Y = y_i|X = \vec{x}_i)\\
    &= \sum_{i=1}^n [\ln \exp(w_{y_i}^T\vec{x}_i) -\ln (1 +\sum_{l=1}^{I-1} \exp(w_l^T\vec{x}_i)) ]\\
    &= \sum_{i=1}^n w_{y_i}^T\vec{x}_i -\ln (1 +\sum_{l=1}^{K-1} \exp(w_l^T\vec{x}_i))
"""
    acc = 0
    for r_idx, xval in X.iterrows():
        w = W[y[r_idx] -1,:] # matrices index-0, but data index-1
        term1 = np.dot(w,xval)
        term2 = np.log(1 +np.sum(np.exp(np.dot(W,xval))))
        acc += (term1 -term2)
    acc += -lda/2. +np.sum(np.power(W,2)) # regularization
    return acc


def gradient(X, y, W, lda):
    """
.. math::
    \grad L(w_k)
    &= \sum_{i=1}^n \left[ I(y_i=k)x_i -\frac{1}{(1 +\sum_{l=1}^{K-1} \exp(w_l^T\vec{x}_i))} \grad_k \sum_{l=1}^{K-1} \exp(w_l^T\vec{x}_i) \right]\\
    &= \sum_{i=1}^n \left[ I(y_i=k)x_i -\frac{1}{(1 +\sum_{l=1}^{K-1} \exp(w_l^T\vec{x}_i))} \exp(w_k^T\vec{x}_i)\vec{x}_i \right]\\
    &= \sum_{i=1}^n \left[ I(y_i=k) -P(y=k|X=x_i) \right]\vec{x}_i\\

weight w_k and indicator y_i must match up, because it is the contribution of each of the x_i to the k'th class.

As for the regularization term,
.. math::
    \grad_k -\frac{\lambda}{2} \sum_{l=1}^{K-1} \norm{w_l}_2^2
    &= -\frac{\lambda}{2} \grad \sum_{l=1}^{K-1} \norm{w_l}_2^2\\
    &= -\frac{\lambda}{2} \grad \norm{w_k}_2^2\\
    &= -\frac{\lambda}{2} 2w_k\\
    &= -\lambda w_k
"""
    n_classes = len(df_train_y.value_counts())
    acc = np.zeros((n_classes,X.shape[1]))
    for r_idx, xval in X.iterrows():
        ind = y[r_idx] # ind is idx-1
        p_y_g_x = p_y_given_x(xval, W) # [P(Y=1|X=x),...,P(Y=K-1|X=x)]
        indicators = np.zeros(n_classes)
        indicators[ind -1] = 1
        acc += np.outer(indicators -p_y_g_x, xval) # one for each weight vector
    acc += -lda * W # regularization
    return acc


def classify(X, W):
    p_ygx = p_Y_given_X(X,W)
    return np.argmax(p_ygx, axis=1) +1 # idx-1


def accuracy(X, y, W):
    ser = (classify(X, W) == y).value_counts()
    return 1.*ser[True]/ser.sum()


@_cache
def gradient_ascent(X_tr, y_tr, X_te, y_te, W_in, eta, max_iter, thres_delta_obj, lda):
    n_classes = len(y_tr.value_counts())
    W = W_in.copy()
    acc_stats = []
    for n_iter in xrange(max_iter):
        ll = log_likelihood(X_tr, y_tr, W, lda)
        accuracy_train = accuracy(X_tr, y_tr, W)
        accuracy_test = accuracy(X_te, y_te, W)
        acc_stats.append( (ll, accuracy_train, accuracy_test) )
        grad = gradient(X_tr, y_tr, W, lda) # dim K,d
        W = W +eta*grad
        if n_iter > 1:
            delta = 1 -1.*acc_stats[-1][0]/acc_stats[-2][0]
            if delta <= thres_delta_obj: # log likelihood improvement threshold
                break
    return W, acc_stats



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-x', help='', default=data_dir +'input/tr_X.txt')
    parser.add_argument('--train-y', help='', default=data_dir +'input/tr_y.txt')
    parser.add_argument('--test-x', help='', default=data_dir +'input/te_X.txt')
    parser.add_argument('--test-y', help='', default=data_dir +'input/te_y.txt')

    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)

    parser.add_argument('--calc', dest='calc', action='store_true')
    parser.set_defaults(calc=False)

    parser.add_argument('--summary', dest='summary', action='store_true')
    parser.set_defaults(summary=False)

    parser.add_argument('--seed', help='', type=int, default=0)

    args = parser.parse_args()

    df_train_X = pd.read_csv(args.train_x, header=None)
    df_train_y = pd.read_csv(args.train_y, header=None)[0] # create a vector, not matrix with 1 column
    df_test_X = pd.read_csv(args.test_x, header=None)
    df_test_y = pd.read_csv(args.test_y, header=None)[0]

    n_classes = len(df_train_y.value_counts())
    n_dims = df_train_X.shape[1] # 256 (16x16)

    eta = 0.0001
    max_iter = 200
    thres_delta_obj = 0.001
    lda = 0
    ldas = [0, 1, 10, 100, 1000]

    def _init_W(n_classes, n_dims):
        seed = args.seed
        np.random.seed(seed)
        W = np.random.random((n_classes, n_dims)) # row-oriented
        return W


    if args.calc:
        for lda in ldas:
            W = _init_W(n_classes, n_dims)

            Wstar, acc_stats = gradient_ascent(df_train_X, df_train_y, df_test_X, df_test_y, W, eta, max_iter, thres_delta_obj, lda)
            df_stats = pd.DataFrame(acc_stats, columns=['ll','tr','te'])
            print lda, df_stats.iloc[-1]


    if args.summary:
        acc = []
        for idx_r, lda in enumerate(ldas):
            W = _init_W(n_classes, n_dims)

            Wstar, acc_stats = gradient_ascent(df_train_X, df_train_y, df_test_X, df_test_y, W, eta, max_iter, thres_delta_obj, lda)
            df_stats = pd.DataFrame(acc_stats, columns=['ll','tr','te'])
            acc.append( [lda] +list(df_stats.iloc[-1]) +[df_stats.iloc[-1].name] )
        print 'max_iter={}, delta_obj={}, eta={}'.format(max_iter, thres_delta_obj, eta)
        print np.round(pd.DataFrame(acc, columns=['lda','ll','tr','te','n']).set_index('lda'), 3)


    if args.plot:
        plt.close()
        fig, axs = plt.subplots(len(ldas), 2, sharex=True)
        min_ll, max_ll = float('inf'), -float('inf')
        for idx_r, lda in enumerate(ldas):
            W = _init_W(n_classes, n_dims)

            Wstar, acc_stats = gradient_ascent(df_train_X, df_train_y, df_test_X, df_test_y, W, eta, max_iter, thres_delta_obj, lda)
            df_stats = pd.DataFrame(acc_stats, columns=['ll','tr','te'])
            axs[idx_r, 0].scatter(df_stats.index, df_stats['ll'], s=1, color='k')
            axs[idx_r, 1].scatter(df_stats.index, df_stats['tr'], s=1, color='g', label='train')
            axs[idx_r, 1].scatter(df_stats.index, df_stats['te'], s=1, color='b', label='test')
            axs[idx_r, 1].set_ylim([0.5,1]) # 0,1 too hard to see
            for ax in axs[idx_r, :]:
                ax.set_xlim([0,max_iter])
                ## format for less visual cluster
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
            axs[idx_r, 0].text(150, -8000, '$\lambda$={}'.format(lda))
            axs[idx_r, 0].set_ylim([-10000,0]) # [-25000,0] to hard to see
            # axs[idx_r, 0].set_yscale('log')
            if idx_r == 0:
                axs[idx_r, 0].set_title('log likelihood')
                axs[idx_r, 1].set_title('accuracy')
            if idx_r == 4:
                axs[idx_r, 1].legend(loc='center right', bbox_to_anchor=(1, 0.6))
            if idx_r +1 == len(ldas):
                axs[idx_r, 0].set_xlabel('num iter')
                axs[idx_r, 1].set_xlabel('num iter')
            # min_ll = min(min_ll, min(df_stats['ll']))
            # max_ll = max(max_ll, max(df_stats['ll']))
        plt.suptitle('change in log likelihood and accuracy for multi-class logistic regression\nlog likelihood stopping criteria')
        plt.savefig('img/hw3_-_k_logreg_-_accuracy,ll.png', format='png')
