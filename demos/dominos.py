## probabilistic dominos

from chain import gen_test_data, calc_xn_x1, calc_x1_xn
import pandas as pd
import numpy as np
from itertools import product

if __name__ == '__main__':
    seed = 0
    kind = '01'
    n_vars = [2, 4, 8, 16, 32, 64, 128, 1024, 2048]
    epsilons = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]

    acc_stats = []
    for n_var, epsilon in product(n_vars, epsilons):
        df_px = gen_test_data(n_var, kind=kind, seed=seed, epsilon=epsilon)
        p_xn_x1, df_stats = calc_xn_x1(df_px)
        p_x1_1_xn_0, p_x1_1_xn_1 = calc_x1_xn(df_px, p_xn_x1)
        acc_stats.append( (n_var, kind, seed, epsilon, p_xn_x1(1), p_xn_x1(0), p_x1_1_xn_1, p_x1_1_xn_0) )

    name_p_vals = ['p(xn=1|x1=1)', 'p(xn=1|x1=0)', 'p(x1=1|xn=1)', 'p(x1=1|xn=0)']
    name_cols = ['n_var', 'kind', 'seed', 'e'] +name_p_vals
    df_stats = pd.DataFrame(acc_stats, columns=name_cols)
    df_piv_stats = pd.pivot_table(df_stats, index='n_var', columns='e', values=name_p_vals).T.to_panel()

    # for n_var in df_piv_stats.items:
    #     print n_var
    #     print df_piv_stats.ix[n_var,:,:]

    print np.round(df_piv_stats.to_frame(), 2)
