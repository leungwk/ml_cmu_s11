import pandas as pd
import random
import numpy as np

data_dir = 'data/hw3/'
cache_dir = 'cache/'

def gen_test_data(n_var, seed=0, **kwargs):
    """generate P(X_{k+1}=1|X_k=j) and P(X_1=1) = P(X_1=1|X_0) for any X_0 (for convenience)"""
    np.random.seed(seed)
    col_names = [0,1]
    kind = kwargs['kind']
    if kind == 'rand':
        df_px = pd.DataFrame(np.random.random((n_var+1,2)), columns=col_names)
    elif kind == 'even':
        df_px = pd.DataFrame(np.zeros((n_var+1,2))+0.5, columns=col_names)
    elif kind == '01': # probablistic dominos
        mtx = np.zeros((n_var+1,2))
        epsilon = kwargs.get('epsilon', None) or 0.01
        mtx[:,0] += epsilon
        mtx[:,1] += (1 -epsilon)
        df_px = pd.DataFrame(mtx, columns=col_names)

    df_px.loc[0,0] = df_px.loc[0,1] # P(X_1=1|X_0=1) <- P(X_1=1|X_0=0) = P(X_1=1)
    df_px.index.name = 'k'
    df_px.columns.name = 'x_k=1'
    return df_px
    

def calc_xn_x1(df_px):
    n = len(df_px) -1 # -1 because "X0" included, not because of idx-0
    if n == 2:
        f = lambda xn2: df_px.loc[n-1,xn2]
        return f, pd.DataFrame()
    #
    f_0 = lambda xn2: (1-df_px.loc[n-1,xn2])*df_px.loc[n,0] +df_px.loc[n-1,xn2]*df_px.loc[n,1]
    # f_0 = lambda xn2: (1-df_px.loc[n-1,xn2])*(1 -df_px.loc[n,0]) +df_px.loc[n-1,xn2]*df_px.loc[n,1]
    f = f_0
    acc_stats = []
    n_i = n-1 # x_{n-2}
    while n_i > 2:
        f0, f1 = f(0), f(1)
        acc_stats.append( (n_i, f0, f1) )
        f = lambda xk: (1-df_px.loc[n_i,xk])*f0 +df_px.loc[n_i,xk]*f1 # where xk means x_{n_i -1}
        n_i -= 1
    ## now f(x_1) = P(X_n=1|X_1=x_1)
    df_stats = pd.DataFrame(acc_stats, columns=['k',0,1]) if acc_stats else pd.DataFrame()
    ## P(X_n=1|X_1=0), P(X_n=1|X_1=1)
    f(0), f(1)
    return f, df_stats


def calc_x1_xn_2(df_px, p_xn_x1):
    """errorneous on some inputs"""
    ## calculate P(X_n=1)
    p_x1_1 = df_px.loc[0,0] # P(X_1=1)

    p_xn_1 = df_px.loc[0,0] # P(X_1=1)
    n = len(df_px) -1 # -1 because "X0" included
    for n_i in xrange(1,n):
        ## P(X_k=1)P(X_{k+1}=1|X_k=1) +P(X_k=0)P(X_{k+1}=1|X_k=0)
        p_xn_1 = p_xn_1*df_px.loc[n_i,1] +(1-p_xn_1)*df_px.loc[n_i,0]
    ## now p_xn_1 = P(X_n=1)
    ## calculate P(X_1=1|X_n=x_n) = \frac{1}{P(X_n=n)}P(X_n=x_n|X_1=1)P(X_1=1)
    p_x1_1_xn_1 = 1./p_xn_1*p_xn_x1(1)*p_x1_1
    p_x1_1_xn_0 = 1./(1-p_xn_1)*(1-p_xn_x1(1))*p_x1_1
    ## f(0) = P(X_n=1|X_1=0), f(1) = P(X_n=1|X_1=1)
    ## errorneous on some inputs
    return p_x1_1_xn_0, p_x1_1_xn_1
    

def calc_x1_xn(df_px, p_xn_x1):
    p_x1_1 = df_px.loc[0,0] # P(X_1=1)
    ## P(X_1=1|X_n=1) = 1/(1 +\frac{P(X_n=1|X_1=0)}{P(X_n=1|X_1=1)}\frac{P(X_1=0)}{P(X_1=1)})
    p_x1_1_xn_1 = 1 / (1 + 1.*(p_xn_x1(0)/p_xn_x1(1))*((1-p_x1_1)/p_x1_1) )
    ## P(X_1=1|X_n=0) = 1/(1 +\frac{P(X_n=0|X_1=0)}{P(X_n=0|X_1=1)}\frac{P(X_1=0)}{P(X_1=1)})
    p_x1_1_xn_0 = 1 / (1 + 1.*((1-p_xn_x1(0))/(1-p_xn_x1(1)))*((1-p_x1_1)/p_x1_1) )
    return p_x1_1_xn_0, p_x1_1_xn_1


if __name__ == '__main__':
    ## run as script to do 1 run
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n-var', help='length of chain', type=int)
    parser.add_argument('--kind', help='kind of model', default='01')
    parser.add_argument('--seed', help='', type=int, default=0)
    parser.add_argument('--epsilon', help='P(X_{k+1}=1|X_k=0) = \\epsilon', type=float)
    args = parser.parse_args()

    n_var, kind, seed = args.n_var, args.kind, args.seed
    df_px = gen_test_data(n_var, kind=kind, seed=seed, epsilon=args.epsilon)

    p_xn_x1, df_stats = calc_xn_x1(df_px)

    # p_x1_1_xn_0, p_x1_1_xn_1 = calc_x1_xn_2(df_px, p_xn_x1)
    p_x1_1_xn_0, p_x1_1_xn_1 = calc_x1_xn(df_px, p_xn_x1)

    print "N = {}".format(n_var)
    print "kind = {}".format(kind)
    print "seed = {}".format(seed)
    print "P(X_n=1|X_1=1) = {}".format(p_xn_x1(1))
    print "P(X_n=1|X_1=0) = {}".format(p_xn_x1(0))
    print "P(X_1=1|X_n=1) = {}".format(p_x1_1_xn_1)
    print "P(X_1=1|X_n=0) = {}".format(p_x1_1_xn_0)
